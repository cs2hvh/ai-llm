#!/usr/bin/env python3
"""Train the MyLLM SentencePiece-Unigram tokenizer (Phase 1).

v2 (2026-05-10) — switched from byte-level BPE to SentencePiece-Unigram per
playbook alignment review:
    - Better cross-lingual quality (used by Llama 1/2, Mistral, Sarvam-1)
    - NFKC normalization for canonical Unicode form
    - Metaspace pre-tokenizer (SentencePiece-style whitespace handling)
    - byte_fallback=True so any UTF-8 sequence round-trips, even without
      a vocab entry — eliminates OOV at inference

Reads ``configs/tokenizer.yaml``, streams samples per the configured corpus
mix, trains the tokenizer, runs validation gates, saves ``tokenizer.json``,
and (with ``--upload``) pushes it to R2 at ``tokenizer/<name>.json``.

Usage:
    python scripts/train_tokenizer.py \\
        --config configs/tokenizer.yaml \\
        --output artifacts/tokenizer.json \\
        --samples-per-source 5_000_000 \\
        --upload
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
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
# Corpus iteration
# --------------------------------------------------------------------------- #
def build_corpus_iterator(
    config: dict,
    samples_per_source: int,
    *,
    max_doc_chars: int = 200_000,
    progress_every: int = 10_000,
) -> Iterator[str]:
    """Stream text strings from the configured corpus mix.

    Uses ``MixtureSampler`` so the long-run share drawn from each source
    matches the configured ``share``.

    Per-doc length cap (``max_doc_chars``) prevents single very-long docs
    (some Wikipedia articles are 1MB+) from inflating the suffix-array
    construction in the SentencePiece-Unigram trainer. The Rust
    ``esaxx-rs`` library used by HuggingFace tokenizers has internal i32
    bounds on substring-position counters; without this cap we observed
    a ``TryFromIntError(())`` panic during the 2026-05-11 production run
    at 1M docs/source.

    Emits a structured log every ``progress_every`` docs so silent multi-
    hour streaming phases don't fly blind.
    """
    sources: dict[str, Iterator[str]] = {}
    weights: list[SourceWeight] = []
    for entry in config["training_corpus"]:
        ds = entry["source"]
        share = float(entry["share"])
        loader = HFStreamLoader(
            dataset=ds,
            category="web",  # category irrelevant for tokenizer training
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
    yielded = 0
    truncated = 0
    total_chars = 0
    for _, text in sampler:
        if not text or len(text) < 50:
            continue
        if len(text) > max_doc_chars:
            text = text[:max_doc_chars]
            truncated += 1
        yielded += 1
        total_chars += len(text)
        if yielded % progress_every == 0:
            log.info(
                "corpus_progress",
                docs=yielded,
                total_chars=total_chars,
                gb_text=round(total_chars / 1e9, 2),
                truncated=truncated,
            )
        yield text
    log.info(
        "corpus_drained",
        docs=yielded,
        total_chars=total_chars,
        gb_text=round(total_chars / 1e9, 2),
        truncated=truncated,
    )


# --------------------------------------------------------------------------- #
# Train
# --------------------------------------------------------------------------- #
def train(
    config: dict,
    samples_per_source: int,
    output_path: Path,
) -> object:
    """Train and save the SentencePiece-Unigram tokenizer.

    Byte-fallback notes (real-world bug we hit in v2 smoke 2026-05-10):
        ``Unigram(byte_fallback=True)`` on the model is silently dropped
        by ``UnigramTrainer`` — the trainer rebuilds the model without it,
        and the saved JSON has ``byte_fallback: false``. To get true
        byte-fallback (and round-trip correctness for unseen scripts) we:
            1. inject the 256 ``<0xXX>`` byte-piece tokens as special
               tokens so they get vocab IDs (and survive trainer pruning),
            2. post-process the saved JSON to flip ``byte_fallback`` to
               ``true``.
        Without this, characters absent from the training data are
        UNK'd and silently dropped on decode (we observed this in
        Chinese and Arabic — see ``artifacts/tokenizer_smoke.log`` v1).
    """
    try:
        from tokenizers import Tokenizer
        from tokenizers.decoders import (
            ByteFallback,
            Fuse,
            Replace,
        )
        from tokenizers.decoders import Sequence as DecoderSequence
        from tokenizers.decoders import Strip as DecoderStrip
        from tokenizers.models import Unigram
        from tokenizers.normalizers import NFKC
        from tokenizers.pre_tokenizers import Metaspace
        from tokenizers.trainers import UnigramTrainer
    except ImportError as e:
        raise ImportError("tokenizers not installed; pip install tokenizers") from e

    algorithm = config.get("algorithm", "sentencepiece_unigram")
    if algorithm != "sentencepiece_unigram":
        raise ValueError(
            f"unsupported algorithm: {algorithm!r}; "
            f"this trainer only supports sentencepiece_unigram"
        )

    # Build special-token list from constants module — single source of truth.
    base_special = all_special_token_strings(
        reserved_slots=int(config.get("reserved_slots", 0))
    )

    # 256 byte-piece pseudo-tokens — required for byte-fallback round-trip.
    byte_pieces = [f"<0x{i:02X}>" for i in range(256)]
    special_tokens = base_special + byte_pieces

    # Unigram requires UNK; locate its index in our special-token order.
    unk_token = SpecialTokens.UNK
    if unk_token not in special_tokens:
        raise RuntimeError(f"{unk_token} missing from special tokens; cannot train Unigram")

    tokenizer = Tokenizer(Unigram())
    tokenizer.normalizer = NFKC()
    pre_replacement = config.get("pre_tokenizer", {}).get("replacement", "▁")
    tokenizer.pre_tokenizer = Metaspace(replacement=pre_replacement, prepend_scheme="always")
    # SentencePiece-style decoder with byte-fallback reassembly:
    #   Replace "▁" → " "        (undo Metaspace whitespace marker)
    #   ByteFallback             (parse <0xXX> tokens into raw bytes)
    #   Fuse                     (concatenate adjacent string pieces)
    #   Strip(left=1)            (remove the leading space Metaspace prepended)
    tokenizer.decoder = DecoderSequence([
        Replace(pre_replacement, " "),
        ByteFallback(),
        Fuse(),
        DecoderStrip(content=" ", left=1, right=0),
    ])

    trainer = UnigramTrainer(
        vocab_size=int(config["vocab_size"]),
        special_tokens=special_tokens,
        unk_token=unk_token,
        show_progress=True,
        max_piece_length=16,
    )

    # Esaxx-rs scale-ceiling warning.
    # Approximate: total chars across all docs must stay well below
    # ~2^31 (~2.1B) to avoid i32 overflow in suffix-array construction.
    # Empirical safe zone observed: total corpus ≤ ~10-15 GB text.
    avg_doc_chars = 5_000  # rough average across our 6 sources
    n_sources = len(config["training_corpus"])
    expected_chars = samples_per_source * n_sources * avg_doc_chars
    if expected_chars > 15_000_000_000:  # >15 GB
        log.warning(
            "corpus_size_near_esaxx_ceiling",
            samples_per_source=samples_per_source,
            n_sources=n_sources,
            expected_gb=round(expected_chars / 1e9, 1),
            note="esaxx-rs may panic at TryFromIntError above ~15GB text; "
                 "consider lower --samples-per-source",
        )

    log.info(
        "tokenizer_train_start",
        algorithm=algorithm,
        vocab_size=config["vocab_size"],
        n_special=len(special_tokens),
        n_byte_pieces=len(byte_pieces),
        sources=[c["source"] for c in config["training_corpus"]],
        samples_per_source=samples_per_source,
        expected_gb=round(expected_chars / 1e9, 1),
    )

    corpus = build_corpus_iterator(config, samples_per_source)
    tokenizer.train_from_iterator(corpus, trainer=trainer)

    log.info(
        "tokenizer_train_done",
        actual_vocab_size=tokenizer.get_vocab_size(),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))

    # Post-process: enable byte_fallback AND demote byte pieces from
    # "special" to plain vocab so ``decode(skip_special_tokens=True)``
    # (the default) doesn't silently drop them. Without this, characters
    # absent from the training vocab encode as <0xXX> byte tokens but
    # are filtered out before the ByteFallback decoder ever sees them.
    import json as _json
    with open(output_path, encoding="utf-8") as f:
        td = _json.load(f)
    td.setdefault("model", {})["byte_fallback"] = True
    n_demoted = 0
    for tok in td.get("added_tokens", []):
        content = tok.get("content", "")
        if content.startswith("<0x") and content.endswith(">") and len(content) == 6:
            tok["special"] = False
            n_demoted += 1
    with open(output_path, "w", encoding="utf-8") as f:
        _json.dump(td, f, ensure_ascii=False)

    # Reload so the returned tokenizer reflects the patched JSON.
    tokenizer = Tokenizer.from_file(str(output_path))

    sha = sha256_file(output_path)
    log.info(
        "tokenizer_saved",
        path=str(output_path),
        sha256=sha,
        byte_fallback=True,
        byte_pieces_demoted=n_demoted,
    )
    return tokenizer


# --------------------------------------------------------------------------- #
# Validate
# --------------------------------------------------------------------------- #
def validate(tokenizer: object, validation_config: dict) -> list[str]:
    """Return list of validation failures. Empty list = pass."""
    failures: list[str] = []

    try:
        verify_tokenizer_has_required(tokenizer)
    except ValueError as e:
        failures.append(str(e))

    if validation_config.get("roundtrip_exactness_required", False):
        for sample in validation_config.get("code_pattern_smoketests", []):
            ids = tokenizer.encode(sample).ids  # type: ignore[attr-defined]
            decoded = tokenizer.decode(ids)  # type: ignore[attr-defined]
            # SentencePiece adds a leading space artifact; allow that or exact match.
            if decoded.lstrip() != sample.lstrip():
                failures.append(
                    f"roundtrip mismatch for {sample!r}: got {decoded!r}"
                )

    return failures


# --------------------------------------------------------------------------- #
# Upload
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
    p.add_argument("--output", default="artifacts/tokenizer.json")
    p.add_argument(
        "--samples-per-source",
        type=int,
        default=5_000_000,
        help="Cap on documents drawn from each source.",
    )
    p.add_argument("--upload", action="store_true")
    p.add_argument("--remote-key", default=None)
    args = p.parse_args()

    configure_logging()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        log.error("config_not_found", path=str(config_path))
        return 2

    with open(config_path) as f:
        config = yaml.safe_load(f)

    output = Path(args.output)
    tokenizer = train(config, args.samples_per_source, output)

    failures = validate(tokenizer, config.get("validation", {}))
    if failures:
        for fmsg in failures:
            log.error("tokenizer_validation_failure", reason=fmsg)
        return 1

    log.info("tokenizer_validation_passed")

    if args.upload:
        remote_key = args.remote_key or f"tokenizer/{config['name']}.json"
        url = upload_to_r2(output, remote_key)
        log.info("tokenizer_uploaded", url=url)

    return 0


if __name__ == "__main__":
    sys.exit(main())
