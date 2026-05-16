"""Benchmark adapters for the eval harness.

Each module here implements the ``Benchmark`` protocol from
``myllm.eval.types`` for one named dataset.

Currently implemented:
    - ``mmlu_pro.MMLUProBenchmark``    — English MMLU-Pro (10-choice MCQ)
    - ``mmlu_prox.MMLUProXBenchmark``  — 29-language MMLU-Pro
    - ``gsm8k.GSM8KBenchmark``         — grade-school math (#### N parse)
    - ``belebele.BelebeleBenchmark``   — 122-language reading comprehension
    - ``milu.MILUBenchmark``           — 11-Indic-language MCQ (default Hindi)

Planned (next PRs):
    - ``humaneval_plus`` / ``mbpp_plus`` — code-exec sandboxes
    - ``ifeval``         — programmatic instruction-following constraints
    - ``global_mmlu``    — multilingual knowledge
"""
from myllm.eval.benchmarks.belebele import BelebeleBenchmark
from myllm.eval.benchmarks.gsm8k import GSM8KBenchmark
from myllm.eval.benchmarks.milu import MILUBenchmark
from myllm.eval.benchmarks.mmlu_pro import MMLUProBenchmark
from myllm.eval.benchmarks.mmlu_prox import MMLUProXBenchmark

__all__ = [
    "BelebeleBenchmark",
    "GSM8KBenchmark",
    "MILUBenchmark",
    "MMLUProBenchmark",
    "MMLUProXBenchmark",
]
