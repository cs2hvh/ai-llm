#!/usr/bin/env python3
"""Per-source packed-corpus build CLI (B2).

Drives ``myllm.data.build.build_one_source`` from a source-config YAML.
One source per invocation — workers launch this in parallel over disjoint
sources (DataTrove-pattern; Slurm/Ray/subprocess-fan-out are all viable).

Output layout per source::

    <output-root>/<source-id>/
        manifest.json
        shard-000000/{tokens.bin, seq_meta.arrow, doc_meta.parquet, manifest.json}
        shard-000001/...

After all sources finish, the mixed-training corpus is composed by
``scripts/compose_mixed_corpus.py`` (which reads each per-source
manifest, validates tokenizer + sequence_length, applies the planned
mixture shares, and writes a single unified shard set).

Usage::

    python scripts/build_packed_corpus.py \\
        --pretrain-mix-config configs/data/pretrain_mix.yaml \\
        --source fineweb-edu \\
        --tokenizer-path artifacts/tokenizer_v1.json \\
        --output-root /data/corpus_v1/sources \\
        --sequence-length 8192 \\
        --sequences-per-shard 65536 \\
        --revision-id 2024-12-05

Defaults are read from configs/base_1b.yaml (sequence_length) and
configs/data/pretrain_mix.yaml (source list, target_share, text_field,
config_name, split, trust_remote_code).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from myllm.data.build import (  # noqa: E402
    FINEWEB_MINHASH_CONFIG,
    BuildStats,
    _tokenizer_sha256,
    build_one_source,
)
from myllm.data.types import Document, DocumentSource  # noqa: E402
from myllm.utils import configure_logging, get_logger  # noqa: E402

log = get_logger(__name__)


def _resolve_source_config(pretrain_mix: dict, source_id: str) -> dict:
    """Find the matching entry in configs/data/pretrain_mix.yaml's `sources`."""
    sources = pretrain_mix.get("sources", [])
    # source_id can match the dataset name (e.g. "HuggingFaceFW/fineweb-edu")
    # OR a short alias if the entry has one. For now match against dataset.
    for entry in sources:
        if entry.get("dataset") == source_id or entry.get("source_id") == source_id:
            return entry
        # Accept short-hand mapping for common sources.
        short = entry.get("dataset", "").split("/")[-1].lower()
        if source_id.lower().replace("-", "_").replace(".", "_") == short.replace("-", "_").replace(".", "_"):
            return entry
    raise ValueError(
        f"source {source_id!r} not found in pretrain_mix; "
        f"available: {[s.get('dataset') for s in sources]}"
    )


def _open_hf_stream(
    source_entry: dict,
    *,
    revision: str | None,
    sample_limit: int | None,
) -> Iterator[Document]:
    """Open the source's HF stream and yield Document objects.

    Lazy import of myllm.data.loader so the CLI can be unit-tested
    without HF datasets installed. The actual streaming/filter wiring
    matches what scripts/run_pretrain.py already uses.
    """
    from myllm.data.loader import HFStreamLoader

    dataset_id = source_entry["dataset"]
    text_field = source_entry.get("text_field", "text")
    category: DocumentSource = source_entry.get("category", "web")
    config_name = source_entry.get("config_name")
    split = source_entry.get("split", "train")
    trust_remote_code = bool(source_entry.get("trust_remote_code", False))

    loader = HFStreamLoader(
        dataset=dataset_id,
        text_field=text_field,
        category=category,
        config_name=config_name,
        split=split,
        trust_remote_code=trust_remote_code,
        sample_limit=sample_limit,
        revision=revision,
    )
    yield from loader


def _build_filter_chain(pretrain_mix: dict):
    """Construct the document filter chain from pretrain_mix.filters block."""
    from myllm.data.filters import (
        FilterChain,
        LengthFilter,
        PIIRedactor,
        RepetitionFilter,
        SymbolRatioFilter,
    )

    filters_cfg = pretrain_mix.get("filters", {})
    chain: list = []

    if "length" in filters_cfg:
        lc = filters_cfg["length"]
        chain.append(LengthFilter(
            min_chars=int(lc.get("min_chars", 200)),
            max_chars=int(lc.get("max_chars", 1_000_000)),
        ))
    if "repetition" in filters_cfg:
        rc = filters_cfg["repetition"]
        chain.append(RepetitionFilter(
            max_top_word_share=float(rc.get("max_top_word_share", 0.20)),
            max_top_ngram_share=float(rc.get("max_top_ngram_share", 0.10)),
            ngram_n=int(rc.get("ngram_n", 5)),
        ))
    if "symbol_ratio" in filters_cfg:
        sc = filters_cfg["symbol_ratio"]
        chain.append(SymbolRatioFilter(
            max_symbol_ratio=float(sc.get("max_symbol_ratio", 0.30)),
        ))
    if "pii" in filters_cfg:
        pc = filters_cfg["pii"]
        chain.append(PIIRedactor(
            redact_email=bool(pc.get("redact_email", True)),
            redact_phone=bool(pc.get("redact_phone", True)),
            redact_ipv4=bool(pc.get("redact_ipv4", False)),
        ))

    if not chain:
        return None
    return FilterChain(tuple(chain))


def _make_filter_callable(filter_chain) -> callable:  # type: ignore[valid-type]
    """Adapt FilterChain to the (doc) -> bool callable build_one_source expects."""
    if filter_chain is None:
        return None

    def _fn(doc: Document) -> bool:
        # FilterChain.apply returns (Document, FilterDecision). The
        # filters mutate doc.text in place for redaction (PII), so we
        # don't need to copy anything back. We just gate on .keep.
        _, decision = filter_chain.apply(doc)
        return decision.keep

    return _fn


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, help="dataset id or short alias")
    p.add_argument("--pretrain-mix-config",
                   default=str(_REPO / "configs" / "data" / "pretrain_mix.yaml"))
    p.add_argument("--model-config",
                   default=str(_REPO / "configs" / "base_1b.yaml"),
                   help="Read sequence_length from here (default base_1b.yaml).")
    p.add_argument("--tokenizer-path",
                   default=str(_REPO / "artifacts" / "tokenizer_v1.json"))
    p.add_argument("--output-root", required=True,
                   help="Per-source output goes under <output-root>/<source-id>/")
    p.add_argument("--source-id", default=None,
                   help="Override the source_id stored in the manifest "
                        "(default: short name from --source)")
    p.add_argument("--sequence-length", type=int, default=None,
                   help="Override the seq_len read from --model-config")
    p.add_argument("--sequences-per-shard", type=int, default=65536,
                   help="Default 65,536 (~256K tokens × 8k seq = 524M tokens/shard).")
    p.add_argument("--revision-id", default="unpinned",
                   help="Label stored in the corpus manifest (free-form: "
                        "an HF commit SHA, a date tag, 'smoke-test', etc.). "
                        "This is the provenance value that ends up in "
                        "DocSpan.dataset_revision_id. It is NOT passed to "
                        "HF's load_dataset — see --hf-revision for that.")
    p.add_argument("--hf-revision", default=None,
                   help="HF Hub revision (commit SHA or branch) passed to "
                        "datasets.load_dataset(revision=...). Default None "
                        "= HF default (usually 'main'). Pass a real SHA "
                        "to pin the dataset version for reproducibility.")
    p.add_argument("--sample-limit", type=int, default=None,
                   help="Cap on docs to PROCESS (pre-filter). For smoke tests.")
    p.add_argument("--target-tokens", type=int, default=None,
                   help="Token-level emission cap. Stops appending once "
                        "this many tokens have been written to disk. Bounds "
                        "wall time when doc sizes vary across sources "
                        "(pg19 books vs FineWeb-Edu pages).")
    p.add_argument("--no-dedupe", action="store_true",
                   help="Disable MinHash+LSH dedupe (per-source). Default: enabled.")
    p.add_argument("--no-filters", action="store_true",
                   help="Skip filter chain (length, repetition, PII). Default: apply.")
    p.add_argument("--no-decontam", action="store_true",
                   help="Skip benchmark decontamination. Default: apply if index_path set in pretrain_mix.")
    p.add_argument("--eos-token-id", type=int, default=2,
                   help="EOS token id (default 2 = SPM convention).")
    p.add_argument("--pad-token-id", type=int, default=0,
                   help="Pad token id (default 0).")
    p.add_argument("--drop-last", action="store_true", default=False,
                   help="Drop the trailing partial sequence (default: keep + pad).")
    p.add_argument("--r2-prefix", default=None,
                   help="If set, mirror each completed shard to "
                        "s3://$S3_BUCKET/<r2-prefix>/<source-id>/shard-NNNNNN/* "
                        "as soon as the shard closes. The bucket comes from "
                        "$S3_BUCKET; the S3-compatible endpoint comes from "
                        "$S3_ENDPOINT_URL.")
    p.add_argument("--delete-local-after-upload", action="store_true",
                   help="With --r2-prefix, delete the local shard files after "
                        "a successful upload. Lets a multi-TB build run on a "
                        "machine with limited local disk (streams to R2).")
    p.add_argument("--production", action="store_true",
                   help="Production mode (2026-05-12 reviewer P0-5 fix): "
                        "fail-closed on (a) decontamination skipped without "
                        "a pre-built index, and (b) --no-decontam unless "
                        "--allow-unsafe-no-decontam is also set. Default "
                        "off, so infra-validation builds still run.")
    p.add_argument("--allow-unsafe-no-decontam", action="store_true",
                   help="Required to combine --production with --no-decontam. "
                        "Use only when explicitly building a corpus that is "
                        "not intended for benchmark-claim training (e.g. the "
                        "current infra_validation corpus).")
    args = p.parse_args()

    configure_logging()
    # Fail-fast on missing R2 credentials when --r2-prefix is set.
    # Without this guard, every shard's upload silently fails (logged at
    # error level) but the build keeps running, and the operator only
    # discovers the problem hours later. Hit by a 2026-05-12 smoke run
    # where .env wasn't sourced.
    if args.r2_prefix is not None:
        missing = [
            v for v in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
                        "S3_BUCKET", "S3_ENDPOINT_URL")
            if not os.environ.get(v)
        ]
        if missing:
            sys.stderr.write(
                f"ERROR: --r2-prefix is set but these env vars are missing: "
                f"{missing}\n"
                f"Source your .env first:\n"
                f"  set -a && source .env && set +a\n"
            )
            return 2
    pretrain_mix = yaml.safe_load(Path(args.pretrain_mix_config).read_text())
    model_cfg = yaml.safe_load(Path(args.model_config).read_text())
    # 2026-05-12 senior review P0-3 fix: default packed sequence length is
    # model_cfg.context_length + 1, NOT context_length. The training loop's
    # `make_input_label_pairs` does a next-token shift on each packed
    # sequence, so a packed sequence of length S becomes input/label of
    # length S-1 each. For the model's RoPE table (precomputed at
    # context_length) to be a perfect fit, the packed sequence must be
    # context_length + 1 long. run_pretrain.py asserts this via the
    # packed_seq_len == reader.sequence_length check; with the previous
    # default of `context_length`, the trainer refused the corpus.
    seq_len = int(
        args.sequence_length if args.sequence_length is not None
        else (model_cfg["context_length"] + 1)
    )

    source_entry = _resolve_source_config(pretrain_mix, args.source)
    source_id = args.source_id or (
        source_entry["dataset"].split("/")[-1].lower().replace(".", "_")
    )

    out_dir = Path(args.output_root).resolve() / source_id

    log.info(
        "build_packed_corpus_start",
        source=args.source,
        source_id=source_id,
        output_dir=str(out_dir),
        sequence_length=seq_len,
        sequences_per_shard=args.sequences_per_shard,
        revision_id=args.revision_id,
    )

    # Tokenizer.
    from myllm.data.tokenize import load_tokenizer
    tokenizer = load_tokenizer(args.tokenizer_path)
    tok_sha = _tokenizer_sha256(args.tokenizer_path)

    # Filters.
    filter_chain = None if args.no_filters else _build_filter_chain(pretrain_mix)
    filter_fn = _make_filter_callable(filter_chain)

    # Decontamination.
    #
    # 2026-05-12 reviewer P0-5 fix: in --production mode, ANY of the
    # following becomes a hard error:
    #   (a) --no-decontam without --allow-unsafe-no-decontam
    #   (b) decontamination.enabled=true in yaml but no index_path AND
    #       no --allow-unsafe-no-decontam
    # Infra-validation / smoke builds (no --production flag) continue to
    # log a warning and skip, as before.
    if args.production and args.no_decontam and not args.allow_unsafe_no_decontam:
        log.error(
            "production_mode_refuses_no_decontam",
            hint="Pass --allow-unsafe-no-decontam if you intend to build a "
                 "corpus that won't be used for benchmark-claim training.",
        )
        return 3

    decontaminator = None
    if not args.no_decontam:
        decon_cfg = pretrain_mix.get("decontamination", {})
        if decon_cfg.get("enabled", False):
            from myllm.data.decontamination import (
                DecontaminationIndex,
                DualModeDecontaminationIndex,
            )
            # Three yaml configurations, in priority order:
            #   1. index_path_primary + index_path_secondary  -> dual-mode
            #   2. index_path                                 -> single-mode (back-compat)
            #   3. (none of the above)                        -> warn / fail
            #
            # Dual-mode = 8-gram (high recall) + 13-gram (high precision)
            # per reviewer Q7 (2026-05-12). Operators can run production
            # with just one mode if cache size is a concern, but the
            # recommended default is dual.
            primary_path = decon_cfg.get("index_path_primary")
            secondary_path = decon_cfg.get("index_path_secondary")
            legacy_path = decon_cfg.get("index_path")

            if primary_path and secondary_path:
                primary = DecontaminationIndex.load_json(primary_path)
                secondary = DecontaminationIndex.load_json(secondary_path)
                decontaminator = DualModeDecontaminationIndex(
                    primary=primary, secondary=secondary,
                )
                log.info(
                    "decontamination_dual_mode_loaded",
                    primary=primary_path,
                    primary_ngram=primary.config.ngram_size,
                    secondary=secondary_path,
                    secondary_ngram=secondary.config.ngram_size,
                )
            elif primary_path or secondary_path:
                # Only one of the dual paths set — load it as a single-mode index.
                only = primary_path or secondary_path
                decontaminator = DecontaminationIndex.load_json(only)
                log.info(
                    "decontamination_single_mode_loaded",
                    path=only,
                    ngram=decontaminator.config.ngram_size,
                    note="dual-mode requested but only one path set",
                )
            elif legacy_path:
                decontaminator = DecontaminationIndex.load_json(legacy_path)
                log.info("decontamination_index_loaded", path=legacy_path)
            else:
                # Same fail-closed gate as above, for the "enabled=true but
                # no index" path. The default smoke/infra warning is
                # preserved; --production upgrades it to an error.
                if args.production and not args.allow_unsafe_no_decontam:
                    log.error(
                        "production_mode_refuses_decontam_without_index",
                        hint="Pre-build the index via "
                             "scripts/build_decontamination_index.py and "
                             "set decontamination.index_path in "
                             "pretrain_mix.yaml; OR pass "
                             "--allow-unsafe-no-decontam.",
                    )
                    return 3
                log.warning(
                    "decontamination_skipped_no_prebuilt_index_for_corpus_build "
                    "(use scripts/build_decontamination_index.py to make one)"
                )

    # Dedupe.
    dedupe_cfg = None if args.no_dedupe else FINEWEB_MINHASH_CONFIG

    # Stream + build.
    docs = _open_hf_stream(
        source_entry,
        revision=args.hf_revision,
        sample_limit=args.sample_limit,
    )
    target_share = float(source_entry.get("share", 0.0))

    t0 = time.time()
    # R2 prefix is per-source: <r2-prefix>/<source-id>/shard-NNNNNN/...
    per_source_r2_prefix = (
        f"{args.r2_prefix.rstrip('/')}/{source_id}"
        if args.r2_prefix else None
    )

    stats, manifest = build_one_source(
        source_id=source_id,
        docs=docs,
        tokenizer=tokenizer,
        output_dir=out_dir,
        sequence_length=seq_len,
        sequences_per_shard=args.sequences_per_shard,
        revision_id=args.revision_id,
        tokenizer_sha256=tok_sha,
        eos_token_id=args.eos_token_id,
        pad_token_id=args.pad_token_id,
        filter_fn=filter_fn,
        decontaminator=decontaminator,
        dedupe_config=dedupe_cfg,
        target_share=target_share,
        drop_last=args.drop_last,
        sample_limit=args.sample_limit,
        target_tokens=args.target_tokens,
        r2_prefix=per_source_r2_prefix,
        delete_local_after_upload=args.delete_local_after_upload,
    )
    wall_sec = time.time() - t0

    summary = {
        "source_id": source_id,
        "output_dir": str(out_dir),
        "tokenizer_sha256": tok_sha,
        "revision_id": args.revision_id,
        "wall_seconds": round(wall_sec, 1),
        "stats": {
            "docs_seen": stats.docs_seen,
            "docs_kept": stats.docs_kept,
            "docs_filtered": stats.docs_filtered,
            "docs_contaminated": stats.docs_contaminated,
            "docs_deduped": stats.docs_deduped,
            "sequences_emitted": stats.sequences_emitted,
            "tokens_emitted": stats.tokens_emitted,
        },
        "manifest": {
            "n_shards": manifest.n_shards,
            "total_sequences": manifest.total_sequences,
            "total_tokens": manifest.total_tokens,
        },
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
