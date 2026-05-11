"""Tests for the generic eval runner.

Uses a minimal mock Benchmark implementation so the runner's logic is
tested without any benchmark-specific I/O.
"""
from __future__ import annotations

from collections.abc import Iterator

import pytest

from myllm.eval.runner import run_benchmark
from myllm.eval.types import Benchmark, EvalExample


class _MockBenchmark:
    """A tiny benchmark of 10 yes/no questions across 2 languages."""

    name = "mock-bench"

    def __init__(self, examples: list[EvalExample]):
        self._examples = examples

    def load_examples(
        self, split: str = "test", sample_size: int | None = None, seed: int = 0
    ) -> Iterator[EvalExample]:
        rows = self._examples
        if sample_size is not None:
            rows = rows[:sample_size]
        yield from rows

    def score(self, prediction: str, example: EvalExample) -> bool:
        return prediction.strip().lower() == example.target_answer.strip().lower()

    def subgroup_key(self, example: EvalExample) -> str:
        return example.metadata.get("language", "n/a")


def _examples_two_langs():
    """3 English (target=yes) + 2 Hindi (target=no) examples."""
    return [
        EvalExample(prompt="q1", target_answer="yes", metadata={"language": "en"}),
        EvalExample(prompt="q2", target_answer="yes", metadata={"language": "en"}),
        EvalExample(prompt="q3", target_answer="yes", metadata={"language": "en"}),
        EvalExample(prompt="q4", target_answer="no",  metadata={"language": "hi"}),
        EvalExample(prompt="q5", target_answer="no",  metadata={"language": "hi"}),
    ]


def test_runner_perfect_score():
    """Predict always returns the gold answer → 100% overall, 100% per-lang."""
    bench = _MockBenchmark(_examples_two_langs())
    predict = lambda prompt: "yes" if prompt in ("q1", "q2", "q3") else "no"  # noqa: E731
    result = run_benchmark(bench, predict)
    assert result.benchmark == "mock-bench"
    assert result.accuracy == 1.0
    assert result.n_correct == 5 and result.n_total == 5
    assert set(result.per_subgroup.keys()) == {"en", "hi"}
    assert result.per_subgroup["en"].accuracy == 1.0
    assert result.per_subgroup["hi"].accuracy == 1.0


def test_runner_zero_score():
    """Predict always wrong → 0% overall."""
    bench = _MockBenchmark(_examples_two_langs())
    predict = lambda _: "MAYBE"  # noqa: E731
    result = run_benchmark(bench, predict)
    assert result.accuracy == 0.0
    assert result.n_correct == 0 and result.n_total == 5


def test_runner_partial_score():
    """Predict right for English, wrong for Hindi → 60% overall, 100% en, 0% hi."""
    bench = _MockBenchmark(_examples_two_langs())
    predict = lambda prompt: "yes"  # noqa: E731
    result = run_benchmark(bench, predict)
    assert result.accuracy == pytest.approx(3 / 5)
    assert result.per_subgroup["en"].accuracy == 1.0
    assert result.per_subgroup["hi"].accuracy == 0.0
    assert result.per_subgroup["en"].n_correct == 3
    assert result.per_subgroup["hi"].n_correct == 0


def test_runner_short_summary_format():
    """short_summary() string format is parsable and includes per-lang."""
    bench = _MockBenchmark(_examples_two_langs())
    predict = lambda prompt: "yes"  # noqa: E731
    result = run_benchmark(bench, predict)
    summary = result.short_summary()
    assert "mock-bench" in summary
    assert "60.0%" in summary
    assert "en=100.0%" in summary
    assert "hi=0.0%" in summary


def test_runner_respects_sample_size():
    """sample_size=2 only evaluates the first 2 examples."""
    bench = _MockBenchmark(_examples_two_langs())
    predict = lambda _: "yes"  # noqa: E731
    result = run_benchmark(bench, predict, sample_size=2)
    assert result.n_total == 2


def test_runner_handles_empty_benchmark():
    """Empty bench should produce a finite (=0) result rather than ZeroDivisionError."""
    bench = _MockBenchmark([])
    result = run_benchmark(bench, lambda _: "")
    assert result.accuracy == 0.0
    assert result.n_correct == 0 and result.n_total == 0
    assert result.per_subgroup == {}


def test_mock_satisfies_benchmark_protocol():
    """Sanity: our mock conforms to the Benchmark structural type."""
    bench = _MockBenchmark([])
    assert isinstance(bench, Benchmark)
