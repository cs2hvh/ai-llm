#!/usr/bin/env python3
"""Train the MyLLM tokenizer using Google's native SentencePiece (Phase 1, production path).

This is the **enterprise-grade** tokenizer training script. Use it instead of
``train_tokenizer.py`` for production runs. The older script trains via HF
``tokenizers`` (Rust) which uses ``esaxx-rs`` for suffix-array construction —
that path panics with ``TryFromIntError`` above ~10-15GB of text because of
internal i32 counter widths. Google's native ``sentencepiece`` C++ binary
uses 64-bit integers throughout and is what Llama/Mistral/Gemma/Sarvam all
use for their tokenizer training step. We then convert the trained SPM
``.model`` into the HF ``tokenizer.json`` format so the rest of the stack
(HF transformers, vLLM, our training loop) works unchanged.

Pipeline:
    1. Stream the configured corpus mix to a single text file on disk.
       (One doc per line; internal newlines → spaces. Per-doc cap at
       ``--max-doc-chars`` so single huge Wikipedia articles don't dominate.
       Resumable: skip the stream step if the corpus file already exists.)
    2. Train SentencePiece-Unigram with our config: 131,072 vocab, NFKC,
       byte_fallback=True, num_threads=all-available, max_sentence_length=8192,
       train_extremely_large_corpus=True.
    3. Convert ``tokenizer.model`` → HF ``tokenizer.json`` via the
       ``LlamaTokenizerFast`` round-trip (the canonical recipe).
    4. Post-process the JSON: demote byte pieces ``<0x00>..<0xFF>`` from
       ``special:true`` to ``special:false`` so ``decode(skip_special_tokens=True)``
       (the default) doesn't filter them out before ByteFallback decoder runs.
       Same fix we applied to the HF-path script for the same root cause.
    5. Validate via the yaml ``code_pattern_smoketests`` round-trip gate.
    6. (optional) Upload to R2 at the configured remote key.

Usage:
    KERAS_BACKEND=jax .venv/bin/python scripts/train_tokenizer_spm.py \\
        --config configs/tokenizer.yaml \\
        --output artifacts/tokenizer_v1.json \\
        --samples-per-source 1000000 \\
        --upload
"""
from __future__ import annotations

import argparse
import hashlib
import json as _json
import os
import shutil
import sys
import time
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from myllm.data.loader import HFStreamLoader  # noqa: E402
from myllm.data.mixture import MixtureSampler, SourceWeight  # noqa: E402
from myllm.data.special_tokens import (  # noqa: E402
    SpecialTokens,
    all_special_token_strings,
    verify_tokenizer_has_required,
)
from myllm.utils import configure_logging, get_logger  # noqa: E402
from myllm.utils.io import sha256_file  # noqa: E402

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Step 1 — Stream corpus to disk
# --------------------------------------------------------------------------- #
def stream_corpus_to_file(
    config: dict,
    samples_per_source: int,
    out_path: Path,
    *,
    max_doc_chars: int = 200_000,
    progress_every: int = 10_000,
) -> dict:
    """Write the configured corpus mix to ``out_path``, one doc per line.

    Internal newlines are collapsed to spaces so SentencePiece sees clean
    line-delimited sentences. Returns a summary dict with doc count, byte
    count, and per-source counts.
    """
    if out_path.exists() and out_path.stat().st_size > 0:
        log.info(
            "corpus_file_exists_skip_stream",
            path=str(out_path),
            size_gb=round(out_path.stat().st_size / 1e9, 2),
        )
        # Read summary if cached
        summary_path = out_path.with_suffix(out_path.suffix + ".summary.json")
        if summary_path.exists():
            return _json.loads(summary_path.read_text())
        # Fall through with minimal summary
        return {
            "docs": -1,
            "total_chars": out_path.stat().st_size,
            "note": "resumed from existing corpus file; counts unavailable",
        }

    sources: dict[str, "object"] = {}
    weights: list[SourceWeight] = []
    for entry in config["training_corpus"]:
        ds = entry["source"]
        share = float(entry["share"])
        loader = HFStreamLoader(
            dataset=ds,
            category="web",
            text_field=entry.get("text_field", "text"),
            config_name=entry.get("config_name"),
            split=entry.get("split", "train"),
            sample_limit=samples_per_source,
            trust_remote_code=bool(entry.get("trust_remote_code", False)),
        )
        sources[ds] = (doc.text for doc in loader if doc.text)
        weights.append(SourceWeight(ds, share))

    sampler = MixtureSampler(
        sources=sources,
        weights=weights,
        seed=42,
        on_exhaust="drop",
    )

    docs = 0
    truncated = 0
    total_chars = 0
    per_source: dict[str, int] = {}
    t0 = time.time()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for src, text in sampler:
            if not text or len(text) < 50:
                continue
            if len(text) > max_doc_chars:
                text = text[:max_doc_chars]
                truncated += 1
            # Replace internal newlines with spaces so SP sees one doc per line.
            line = text.replace("\n", " ").replace("\r", " ")
            f.write(line)
            f.write("\n")
            docs += 1
            total_chars += len(line)
            per_source[src] = per_source.get(src, 0) + 1
            if docs % progress_every == 0:
                elapsed = time.time() - t0
                log.info(
                    "corpus_stream_progress",
                    docs=docs,
                    gb_written=round(total_chars / 1e9, 2),
                    docs_per_sec=round(docs / max(elapsed, 1e-3), 1),
                    truncated=truncated,
                    elapsed_min=round(elapsed / 60, 1),
                )

    tmp_path.rename(out_path)
    summary = {
        "docs": docs,
        "total_chars": total_chars,
        "gb_written": round(total_chars / 1e9, 2),
        "truncated_docs": truncated,
        "per_source": per_source,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    summary_path = out_path.with_suffix(out_path.suffix + ".summary.json")
    summary_path.write_text(_json.dumps(summary, indent=2))
    log.info("corpus_stream_done", **summary)
    return summary


# --------------------------------------------------------------------------- #
# Step 2 — Train SentencePiece
# --------------------------------------------------------------------------- #
def train_sentencepiece(
    corpus_path: Path,
    model_prefix: Path,
    *,
    vocab_size: int,
    num_threads: int,
    user_defined_symbols: list[str],
    input_sentence_size: int = 2_000_000,
) -> None:
    """Run SentencePiece-Unigram training. Outputs ``<prefix>.model`` + ``<prefix>.vocab``.

    ``input_sentence_size`` is the critical memory knob. With it at 0 (use-all),
    SP loads the full corpus and builds a suffix array proportional to total
    chars — RAM spikes to ~5-8× corpus size. The 2026-05-11 attempt at 28 GB /
    5.99M sentences with input_sentence_size=0 was OOM-killed at +50 min having
    consumed 260 GB RSS on a 251 GB box. With 2M, SP randomly samples 2M of
    6M sentences (shuffle_input_sentence=True preserves source balance),
    capping peak at ~120-150 GB. Llama-2/Mistral train at similar effective
    scale; the 131K vocab quality plateau sits at ~5-15 GB of training text.
    """
    import sentencepiece as spm

    if (model_prefix.parent / (model_prefix.name + ".model")).exists():
        log.info("spm_model_exists_skip_train", prefix=str(model_prefix))
        return

    args = dict(
        input=str(corpus_path),
        model_prefix=str(model_prefix),
        model_type="unigram",
        vocab_size=vocab_size,
        character_coverage=0.9999,        # cover 99.99% of chars; rest go to byte-fallback
        # Special-token IDs at the front of the vocab (must match our SpecialTokens order).
        pad_id=2, pad_piece=SpecialTokens.PAD,
        unk_id=3, unk_piece=SpecialTokens.UNK,
        bos_id=0, bos_piece=SpecialTokens.BOS,
        eos_id=1, eos_piece=SpecialTokens.EOS,
        user_defined_symbols=user_defined_symbols,
        byte_fallback=True,                # add 256 <0xXX> pieces
        normalization_rule_name="nfkc",
        add_dummy_prefix=True,             # Metaspace-style leading-space marker
        treat_whitespace_as_suffix=False,
        split_by_unicode_script=True,
        split_by_number=True,
        split_by_whitespace=True,
        split_digits=True,
        max_sentencepiece_length=16,
        max_sentence_length=16384,         # bigger than default 4192; Wikipedia paragraphs
        num_sub_iterations=2,
        num_threads=num_threads,
        shuffle_input_sentence=True,
        input_sentence_size=input_sentence_size,
        train_extremely_large_corpus=True, # enables 64-bit code path for >10GB
        seed_sentencepiece_size=1_000_000,
        shrinking_factor=0.75,
    )
    log.info("spm_train_start", **{k: v for k, v in args.items() if k != "user_defined_symbols"})
    t0 = time.time()
    spm.SentencePieceTrainer.train(**args)
    log.info(
        "spm_train_done",
        elapsed_min=round((time.time() - t0) / 60, 1),
        model_file=str(model_prefix) + ".model",
        vocab_file=str(model_prefix) + ".vocab",
    )


# --------------------------------------------------------------------------- #
# Step 3 — Convert SPM .model -> HF tokenizer.json
# --------------------------------------------------------------------------- #
def convert_spm_to_hf(model_path: Path, hf_output_dir: Path) -> Path:
    """Use the LlamaTokenizerFast recipe to convert SP -> HF JSON.

    This is the canonical path used by Llama/Mistral/Gemma. Produces a
    full HF tokenizer dir; we'll pull just the ``tokenizer.json`` out.
    """
    # transformers complains about missing torch/tf but tokenizer ops still work.
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    from transformers import LlamaTokenizerFast

    hf_output_dir.mkdir(parents=True, exist_ok=True)
    log.info("spm_to_hf_convert_start", spm_model=str(model_path), out_dir=str(hf_output_dir))
    fast = LlamaTokenizerFast(
        vocab_file=str(model_path),
        legacy=False,
        # Tell HF not to inject Llama-specific BOS/EOS post-processors; we manage
        # those at training/inference time, not in the tokenizer itself.
        add_bos_token=False,
        add_eos_token=False,
    )
    fast.save_pretrained(str(hf_output_dir))
    tj = hf_output_dir / "tokenizer.json"
    if not tj.exists():
        raise RuntimeError(f"conversion produced no tokenizer.json at {tj}")
    log.info("spm_to_hf_convert_done", tokenizer_json=str(tj), size_mb=round(tj.stat().st_size / 1e6, 2))
    return tj


# --------------------------------------------------------------------------- #
# Step 4 — Post-process the HF JSON (byte-piece special-flag fix)
# --------------------------------------------------------------------------- #
def postprocess_hf_json(hf_json_path: Path) -> dict:
    """Demote byte pieces from special:true to special:false.

    Without this, the default ``decode(skip_special_tokens=True)`` filters
    out byte tokens before the ByteFallback decoder reassembles them →
    silent character drop on round-trip for unseen scripts. Same bug we
    caught and fixed in the HF-tokenizers path on 2026-05-10.
    """
    with open(hf_json_path, encoding="utf-8") as f:
        td = _json.load(f)
    n_demoted = 0
    for tok in td.get("added_tokens", []):
        c = tok.get("content", "")
        if c.startswith("<0x") and c.endswith(">") and len(c) == 6:
            if tok.get("special"):
                tok["special"] = False
                n_demoted += 1
    # Ensure byte_fallback is set on the model (SPM converter usually does this,
    # but belt-and-braces in case it slips).
    td.setdefault("model", {})["byte_fallback"] = True
    with open(hf_json_path, "w", encoding="utf-8") as f:
        _json.dump(td, f, ensure_ascii=False)
    log.info("hf_json_postprocessed", byte_pieces_demoted=n_demoted, byte_fallback=True)
    return {"byte_pieces_demoted": n_demoted, "byte_fallback": True}


# --------------------------------------------------------------------------- #
# Step 5 — Validate
# --------------------------------------------------------------------------- #
def validate(hf_json_path: Path, validation_config: dict) -> list[str]:
    from tokenizers import Tokenizer

    failures: list[str] = []
    tok = Tokenizer.from_file(str(hf_json_path))

    try:
        verify_tokenizer_has_required(tok)
    except ValueError as e:
        failures.append(str(e))

    if validation_config.get("roundtrip_exactness_required", False):
        for sample in validation_config.get("code_pattern_smoketests", []):
            ids = tok.encode(sample).ids
            decoded = tok.decode(ids)
            if decoded.lstrip() != sample.lstrip():
                failures.append(
                    f"roundtrip mismatch for {sample!r}: got {decoded!r}"
                )

    # Per-language compression smoke (rough; just logs, doesn't gate).
    samples_for_compression = {
        "en":      "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.",
        "code":    "def fibonacci(n):\n    if n < 2: return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        "hi":      "भारत एक विशाल देश है जिसकी संस्कृति बहुत समृद्ध है।",
        "zh":      "人工智能是一门具有挑战性和创造性的学科。",
        "ar":      "الذكاء الاصطناعي يغير العالم بسرعة كبيرة.",
        "fr":      "L'intelligence artificielle change le monde rapidement.",
        "de":      "Künstliche Intelligenz verändert die Welt rasend schnell.",
        "es":      "La inteligencia artificial está cambiando el mundo rápidamente.",
    }
    compression: dict[str, dict] = {}
    for lang, text in samples_for_compression.items():
        ids = tok.encode(text).ids
        nbytes = len(text.encode("utf-8"))
        compression[lang] = {
            "chars": len(text),
            "bytes": nbytes,
            "tokens": len(ids),
            "bytes_per_token": round(nbytes / max(1, len(ids)), 2),
        }
    log.info("tokenizer_compression_summary", **compression)
    return failures


# --------------------------------------------------------------------------- #
# Step 6 — Upload
# --------------------------------------------------------------------------- #
def upload_to_r2(local_path: Path, remote_key: str) -> str:
    from myllm.utils.storage import upload_file
    return upload_file(local_path, remote_key)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/tokenizer.yaml")
    p.add_argument("--output", default="artifacts/tokenizer_v1.json")
    p.add_argument("--samples-per-source", type=int, default=1_000_000)
    p.add_argument("--max-doc-chars", type=int, default=200_000)
    p.add_argument("--num-threads", type=int, default=os.cpu_count() or 16)
    p.add_argument("--vocab-size", type=int, default=None,
                   help="Override the yaml vocab_size (use for small-corpus smokes).")
    p.add_argument("--input-sentence-size", type=int, default=2_000_000,
                   help="Max sentences SP loads in memory (0 = no cap → will OOM on >15GB corpora).")
    p.add_argument("--corpus-file", default="artifacts/corpus.txt",
                   help="Where to stage the on-disk corpus. Skipped if exists.")
    p.add_argument("--spm-prefix", default="artifacts/tokenizer_spm",
                   help="Prefix for SP model artifacts (.model + .vocab).")
    p.add_argument("--upload", action="store_true")
    p.add_argument("--remote-key", default=None)
    p.add_argument("--keep-intermediates", action="store_true",
                   help="Don't delete corpus.txt / SP artifacts after success.")
    args = p.parse_args()

    configure_logging()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        log.error("config_not_found", path=str(config_path))
        return 2
    config = yaml.safe_load(config_path.read_text())

    # Build the user_defined_symbols list: everything from SpecialTokens
    # except BOS/EOS/PAD/UNK (those go into the canonical slots).
    base_specials = all_special_token_strings(
        reserved_slots=int(config.get("reserved_slots", 0))
    )
    canonical = {SpecialTokens.BOS, SpecialTokens.EOS, SpecialTokens.PAD, SpecialTokens.UNK}
    user_defined = [t for t in base_specials if t not in canonical]

    corpus_path = Path(args.corpus_file)
    spm_prefix = Path(args.spm_prefix)
    hf_out_dir = Path(args.output).parent / (Path(args.output).stem + "_hfdir")
    hf_json_target = Path(args.output)

    # 1. Stream corpus
    stream_summary = stream_corpus_to_file(
        config,
        samples_per_source=args.samples_per_source,
        out_path=corpus_path,
        max_doc_chars=args.max_doc_chars,
    )

    # 2. Train SP
    vocab_size = args.vocab_size if args.vocab_size is not None else int(config["vocab_size"])
    train_sentencepiece(
        corpus_path=corpus_path,
        model_prefix=spm_prefix,
        vocab_size=vocab_size,
        num_threads=args.num_threads,
        user_defined_symbols=user_defined,
        input_sentence_size=args.input_sentence_size,
    )

    # 3. Convert
    converted_tj = convert_spm_to_hf(
        model_path=Path(str(spm_prefix) + ".model"),
        hf_output_dir=hf_out_dir,
    )

    # Move converted tokenizer.json to the final target path.
    hf_json_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(converted_tj, hf_json_target)

    # 4. Post-process
    postprocess_hf_json(hf_json_target)

    # 5. Validate
    failures = validate(hf_json_target, config.get("validation", {}))
    if failures:
        for f in failures:
            log.error("tokenizer_validation_failure", reason=f)
        return 1
    log.info("tokenizer_validation_passed")

    sha = sha256_file(hf_json_target)
    log.info(
        "tokenizer_saved",
        path=str(hf_json_target),
        sha256=sha,
        size_mb=round(hf_json_target.stat().st_size / 1e6, 2),
    )

    # 6. Upload (optional)
    if args.upload:
        remote_key = args.remote_key or f"tokenizer/{config['name']}.json"
        url = upload_to_r2(hf_json_target, remote_key)
        log.info("tokenizer_uploaded", url=url, sha256=sha)

    # 7. Cleanup intermediates (unless asked to keep).
    if not args.keep_intermediates:
        try:
            corpus_path.unlink()
            for ext in (".model", ".vocab"):
                p = Path(str(spm_prefix) + ext)
                if p.exists():
                    p.unlink()
            log.info("intermediates_cleaned", corpus=str(corpus_path), prefix=str(spm_prefix))
        except OSError as e:
            log.warning("cleanup_partial", error=str(e))

    return 0


if __name__ == "__main__":
    sys.exit(main())
