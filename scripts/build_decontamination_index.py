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
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from myllm.data.decontamination import (  # noqa: E402
    DecontaminationConfig,
    DecontaminationIndex,
    extract_prompts_from_benchmark,
)
from myllm.data.prompt_loaders import PROMPT_LOADERS, load_prompts  # noqa: E402
from myllm.utils import configure_logging, get_logger  # noqa: E402

log = get_logger(__name__)


# Benchmarks served by the heavyweight Benchmark adapter in src/myllm/eval/.
# Everything else is served by the lightweight prompt-only loaders in
# src/myllm/data/prompt_loaders.py.
_BENCHMARK_ADAPTER_IDS = {"mmlu-prox", "belebele", "milu"}


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
        f"unknown benchmark id: {bench_id!r}. "
        f"Known adapters: {sorted(_BENCHMARK_ADAPTER_IDS)}"
    )


_DEFAULT_BENCHMARKS = (
    # Existing 3 gate benchmarks (multilingual MCQs via Benchmark adapters)
    "mmlu-prox",
    "belebele",
    "milu",
    # Extended gate set added 2026-05-12 per v1 model card
    "mmlu-pro",
    "humaneval-plus",
    "mbpp-plus",
    "gsm8k",
    "math",
    "mgsm",
    "bbh",
    "ifeval",
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
        default=list(_DEFAULT_BENCHMARKS),
        help=(
            "Benchmark ids to include. Default covers the v1 gate set: "
            f"{', '.join(_DEFAULT_BENCHMARKS)}."
        ),
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
        if bench_id in _BENCHMARK_ADAPTER_IDS:
            # Full Benchmark adapter — used for multilingual MCQ benches
            # where the contamination-relevant text is the formatted
            # prompt (question + choices) that the adapter produces.
            bench = _instantiate_benchmark(bench_id)
            prompts = extract_prompts_from_benchmark(
                bench, split=args.split, sample_size=args.sample_size, seed=args.seed
            )
        elif bench_id in PROMPT_LOADERS:
            # Lightweight prompt-only loader (most v1 gate benchmarks).
            # Each loader knows its own default split — only override if
            # the user passed a non-default --split.
            loader_kwargs: dict[str, Any] = {"sample_size": args.sample_size}
            if args.split != "test":
                loader_kwargs["split"] = args.split
            prompts = load_prompts(bench_id, **loader_kwargs)
        else:
            raise ValueError(
                f"unknown benchmark id: {bench_id!r}. "
                f"Known adapters: {sorted(_BENCHMARK_ADAPTER_IDS)}; "
                f"known prompt loaders: {sorted(PROMPT_LOADERS)}"
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
