#!/usr/bin/env python3
"""Compose per-source packed corpora into a single training-time mixed corpus.

Reads N per-source corpora built by ``build_packed_corpus.py``, interleaves
their packed sequences at the target token shares from
``configs/data/pretrain_mix.yaml``, and emits a single mixed corpus whose
sequence_id space is what the training loop consumes.

Usage::

    python scripts/compose_mixed_corpus.py \\
        --sources-root /data/corpus_v1/sources \\
        --output-dir /data/corpus_v1/train \\
        --pretrain-mix-config configs/data/pretrain_mix.yaml \\
        --sequences-per-shard 65536 \\
        --target-total-sequences 1000000   # ~8B tokens at seq_len=8192

Per-source corpora must be under ``--sources-root/<source-id>/`` with
their own complete top-level ``manifest.json``. Sources missing from the
filesystem are skipped with a loud warning (the operator likely meant to
build them first).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from myllm.data.compose import compose_mixed_corpus  # noqa: E402
from myllm.utils import configure_logging, get_logger  # noqa: E402

log = get_logger(__name__)


def _resolve_sources(
    pretrain_mix: dict, sources_root: Path
) -> tuple[dict[str, Path], dict[str, float]]:
    """Map source entries → existing corpus paths + target shares."""
    sources_map: dict[str, Path] = {}
    target_shares: dict[str, float] = {}
    skipped: list[str] = []

    for entry in pretrain_mix.get("sources", []):
        dataset = entry["dataset"]
        share = float(entry["share"])
        short = dataset.split("/")[-1].lower().replace(".", "_")
        # Per-source build dir convention from build_packed_corpus.py
        candidate = sources_root / short
        if not (candidate / "manifest.json").exists():
            # Try the dataset name as-is (with / replaced by __).
            candidate2 = sources_root / dataset.replace("/", "__")
            if (candidate2 / "manifest.json").exists():
                candidate = candidate2
            else:
                log.warning(
                    "compose_source_missing",
                    dataset=dataset,
                    expected_path=str(candidate),
                )
                skipped.append(dataset)
                continue
        sources_map[short] = candidate
        target_shares[short] = share

    # Renormalize target shares over the present sources so they sum to 1.
    total = sum(target_shares.values())
    if total <= 0:
        raise SystemExit(
            f"no source corpora found under {sources_root}. Run "
            f"scripts/build_packed_corpus.py per source first."
        )
    target_shares = {k: v / total for k, v in target_shares.items()}

    if skipped:
        log.warning(
            "compose_skipped_sources",
            n=len(skipped),
            renormalized=True,
            sources=skipped,
        )

    return sources_map, target_shares


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sources-root", required=True,
                   help="Directory containing per-source corpora "
                        "(<sources-root>/<source-id>/manifest.json must exist).")
    p.add_argument("--output-dir", required=True,
                   help="Where to write the composed training corpus.")
    p.add_argument("--pretrain-mix-config",
                   default=str(_REPO / "configs" / "data" / "pretrain_mix.yaml"),
                   help="Target shares come from this yaml's sources list.")
    p.add_argument("--sequences-per-shard", type=int, default=65536,
                   help="Output shard size in sequences (default 65,536).")
    p.add_argument("--target-total-sequences", type=int, default=None,
                   help="Stop after this many output sequences. None = emit "
                        "until at least one source exhausts.")
    p.add_argument("--corpus-name", default="myllm-train-v1",
                   help="Label stored in the output manifest.")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for the bootstrap-phase weighted draws.")
    p.add_argument("--bootstrap-steps", type=int, default=16,
                   help="Number of weighted picks before deficit-driven mode.")
    args = p.parse_args()

    configure_logging()
    pretrain_mix = yaml.safe_load(Path(args.pretrain_mix_config).read_text())
    sources_root = Path(args.sources_root).resolve()
    output_dir = Path(args.output_dir).resolve()

    sources_map, target_shares = _resolve_sources(pretrain_mix, sources_root)
    log.info(
        "compose_resolved",
        n_sources=len(sources_map),
        sources=sorted(sources_map),
        target_shares=target_shares,
    )

    t0 = time.time()
    stats, manifest = compose_mixed_corpus(
        sources_map,
        output_dir,
        target_shares=target_shares,
        sequences_per_shard=args.sequences_per_shard,
        corpus_name=args.corpus_name,
        target_total_sequences=args.target_total_sequences,
        seed=args.seed,
        bootstrap_steps=args.bootstrap_steps,
    )
    wall_sec = time.time() - t0

    summary = {
        "output_dir": str(output_dir),
        "wall_seconds": round(wall_sec, 1),
        "total_sequences": stats.total_sequences_emitted,
        "total_tokens": stats.total_tokens_emitted,
        "per_source_sequences": stats.per_source_sequences,
        "per_source_tokens": stats.per_source_tokens,
        "actual_source_share": manifest.actual_source_share,
        "target_source_share": manifest.target_source_share,
        "exhausted_sources": stats.exhausted_sources,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
