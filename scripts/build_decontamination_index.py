#!/usr/bin/env python3
"""Offline builder for the R6 benchmark-decontamination index.

The live-build path in ``run_pretrain.py`` works but pulls HF data for
every benchmark at every run-start (minutes). For production runs, build
the index once with this script and point ``decontamination.index_path``
at the resulting JSON.

Usage:
    python scripts/build_decontamination_index.py \\
        --output artifacts/decontamination_index.json \\
        --ngram-size 13 \\
        --sample-size 200 \\
        --benchmarks mmlu-prox belebele milu

The output is a portable JSON file containing the config + per-benchmark
sets of xxhash64 n-gram hashes. It's safe to upload to R2 alongside the
tokenizer artifact and pull at run-start.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from myllm.data.decontamination import (  # noqa: E402
    DecontaminationConfig,
    DecontaminationIndex,
    extract_prompts_from_benchmark,
)
from myllm.utils import configure_logging, get_logger  # noqa: E402

log = get_logger(__name__)


def _instantiate_benchmark(bench_id: str):
    """Mirror of run_pretrain.py's registry; kept here to avoid circular
    imports when run_pretrain itself depends on heavy training deps."""
    if bench_id == "mmlu-prox":
        from myllm.eval.benchmarks import MMLUProXBenchmark
        return MMLUProXBenchmark()
    if bench_id == "belebele":
        from myllm.eval.benchmarks import BelebeleBenchmark
        return BelebeleBenchmark()
    if bench_id == "milu":
        from myllm.eval.benchmarks import MILUBenchmark
        return MILUBenchmark()
    raise ValueError(
        f"unknown benchmark id: {bench_id!r}. Known: mmlu-prox, belebele, milu."
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output",
        required=True,
        help="Path to write the JSON index to.",
    )
    p.add_argument(
        "--benchmarks",
        nargs="+",
        default=["mmlu-prox", "belebele", "milu"],
        help="Benchmark ids to include (default: all three gate benchmarks).",
    )
    p.add_argument(
        "--ngram-size",
        type=int,
        default=13,
        help="N-gram size (default 13, Llama-2/OLMo-2 convention).",
    )
    p.add_argument(
        "--hash-seed",
        type=lambda x: int(x, 0),
        default=0xDECAF,
        help="xxhash64 seed (default 0xDECAF). Must match runtime config.",
    )
    p.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Per-benchmark cap on prompts pulled (default: all).",
    )
    p.add_argument("--split", default="test")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    configure_logging()
    cfg = DecontaminationConfig(
        ngram_size=args.ngram_size,
        hash_seed=args.hash_seed,
    )
    idx = DecontaminationIndex(cfg)

    for bench_id in args.benchmarks:
        log.info("indexing_benchmark", id=bench_id, sample_size=args.sample_size)
        bench = _instantiate_benchmark(bench_id)
        prompts = extract_prompts_from_benchmark(
            bench, split=args.split, sample_size=args.sample_size, seed=args.seed
        )
        idx.add_benchmark(bench_id, prompts)
        sig = idx.signatures[bench_id]
        log.info(
            "benchmark_indexed",
            id=bench_id,
            n_examples=sig.n_examples,
            n_ngrams=len(sig.ngrams),
        )

    idx.save_json(args.output)
    total_ngrams = sum(len(s.ngrams) for s in idx.signatures.values())
    log.info(
        "decontamination_index_saved",
        path=args.output,
        n_benchmarks=len(idx.signatures),
        total_ngrams=total_ngrams,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
