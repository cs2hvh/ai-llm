"""Core data types for the eval harness.

Designed to be backend-agnostic — a ``Benchmark`` doesn't know whether
the model is a real TransformerLM or a mock callable. The runner glues
the two together.

R8 from the 2026-05-11 dossier — adds first-class multilingual eval
support (MMLU-ProX, Belebele, Global-MMLU, MILU) on top of whatever
lm-eval-harness coverage we eventually wire in.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable


@dataclass(frozen=True)
class EvalExample:
    """One question/answer pair in a benchmark.

    ``metadata`` carries the per-example fields a benchmark uses to
    stratify results (language code, subject area, difficulty, etc.).
    """

    prompt: str
    target_answer: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalResult:
    """Outcome of running a model on a benchmark.

    ``per_subgroup`` is a dict from subgroup-key → ``EvalResult`` (e.g.
    ``{"en": EvalResult(...), "hi": EvalResult(...)}`` for a multilingual
    eval). Empty when the benchmark has no subgroups.
    """

    benchmark: str
    accuracy: float
    n_correct: int
    n_total: int
    per_subgroup: dict[str, "EvalResult"] = field(default_factory=dict)

    def short_summary(self) -> str:
        line = f"{self.benchmark}: {self.accuracy*100:5.1f}%  ({self.n_correct}/{self.n_total})"
        if self.per_subgroup:
            sub = "  ".join(
                f"{k}={r.accuracy*100:.1f}%" for k, r in sorted(self.per_subgroup.items())
            )
            line += f"  [{sub}]"
        return line


@runtime_checkable
class Benchmark(Protocol):
    """Adapter contract every benchmark must satisfy.

    A Benchmark is a stateless object describing **how** to evaluate
    against a particular dataset:

      - ``name`` is the public identifier ("mmlu-prox", "belebele", ...).
      - ``load_examples`` returns an iterator of (prompt, target) pairs.
      - ``score`` decides whether a prediction matches the target. Most
        benchmarks use exact-match on the extracted answer; some use
        normalized accuracy.
      - ``subgroup_key`` returns the subgroup string for an example so
        the runner can stratify results (typically language code).
    """

    name: str

    def load_examples(
        self,
        split: str = "test",
        sample_size: int | None = None,
        seed: int = 0,
    ) -> Iterator[EvalExample]: ...

    def score(self, prediction: str, example: EvalExample) -> bool: ...

    def subgroup_key(self, example: EvalExample) -> str: ...


# A "predict function" is anything that takes a prompt and returns a string.
# At eval time this is wired to the trained model's generation path; in
# tests it's a small mock.
PredictFn = Callable[[str], str]
