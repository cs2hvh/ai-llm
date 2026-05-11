"""Benchmark adapters for the eval harness.

Each module here implements the ``Benchmark`` protocol from
``myllm.eval.types`` for one named dataset.

Currently implemented:
    - ``mmlu_prox.MMLUProXBenchmark``  — 29-language MMLU-Pro
    - ``belebele.BelebeleBenchmark``   — 122-language reading comprehension
    - ``milu.MILUBenchmark``           — 11-Indic-language MCQ (default Hindi)

Planned (next PR):
    - ``global_mmlu.GlobalMMLUBenchmark`` — multilingual knowledge
"""
from myllm.eval.benchmarks.belebele import BelebeleBenchmark
from myllm.eval.benchmarks.milu import MILUBenchmark
from myllm.eval.benchmarks.mmlu_prox import MMLUProXBenchmark

__all__ = ["BelebeleBenchmark", "MILUBenchmark", "MMLUProXBenchmark"]
