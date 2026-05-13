"""Tests for the release scorecard machinery (src/myllm/eval/release_scorecard.py).

Pure-Python; no model, no JAX, no checkpoint. Uses tiny mock Benchmark
adapters + a deterministic predict_fn.
"""
from __future__ import annotations

import json
from collections.abc import Iterator

import pytest

from myllm.eval.release_scorecard import (
    BenchmarkScore,
    Scorecard,
    build_scorecard,
    write_scorecard,
)
from myllm.eval.types import EvalExample


# --------------------------------------------------------------------------- #
# Mock Benchmark adapters
# --------------------------------------------------------------------------- #
class _MockMCQ:
    """Always-A multiple choice. score() is exact match on first letter."""
    name = "mock-mcq"

    def __init__(self, target_letters: list[str], languages: list[str] | None = None):
        self._targets = target_letters
        self._languages = languages

    def load_examples(self, split="test", sample_size=None, seed=0) -> Iterator[EvalExample]:
        for i, t in enumerate(self._targets):
            lang = self._languages[i] if self._languages else "en"
            yield EvalExample(prompt=f"Q{i}: pick the right answer Answer:",
                              target_answer=t, metadata={"language": lang})

    def score(self, prediction: str, example: EvalExample) -> bool:
        return bool(prediction) and prediction.strip()[:1].upper() == example.target_answer

    def subgroup_key(self, example: EvalExample) -> str:
        return example.metadata.get("language", "all")


class _MockExploding:
    """A benchmark that raises during load_examples."""
    name = "mock-explode"

    def load_examples(self, split="test", sample_size=None, seed=0):
        raise RuntimeError("HF rate-limited")

    def score(self, prediction, example):
        return False

    def subgroup_key(self, example):
        return "all"


# --------------------------------------------------------------------------- #
# BenchmarkScore / Scorecard format
# --------------------------------------------------------------------------- #
class TestScorecardJSON:
    def test_to_json_includes_all_fields(self):
        card = Scorecard(
            model_checkpoint="ckpt/step-1000",
            model_name="myllm-pilot-250m",
            eval_timestamp_utc="2026-05-13T18:00:00Z",
            sample_size_per_benchmark=50,
            seed=0,
            scores=[
                BenchmarkScore(benchmark="mock", accuracy=0.30, n_total=50,
                              per_subgroup={"en": 0.32, "hi": 0.28}),
            ],
            notes="pilot eval",
        )
        d = json.loads(card.to_json())
        assert d["model_checkpoint"] == "ckpt/step-1000"
        assert d["model_name"] == "myllm-pilot-250m"
        assert d["sample_size_per_benchmark"] == 50
        assert d["scores"][0]["accuracy"] == 0.30
        assert d["scores"][0]["per_subgroup"] == {"en": 0.32, "hi": 0.28}
        assert d["notes"] == "pilot eval"

    def test_to_markdown_renders_headers_and_table(self):
        card = Scorecard(
            model_checkpoint="ckpt/step-5000",
            model_name="myllm-pilot-250m",
            eval_timestamp_utc="2026-05-13T18:00:00Z",
            sample_size_per_benchmark=200,
            seed=42,
            scores=[
                BenchmarkScore(benchmark="mmlu-pro", accuracy=0.27, n_total=200),
                BenchmarkScore(benchmark="gsm8k", accuracy=0.05, n_total=200),
                BenchmarkScore(benchmark="bbh", accuracy=None, n_total=0, error="OOM"),
            ],
        )
        md = card.to_markdown()
        assert "# Release Scorecard — myllm-pilot-250m" in md
        assert "**Checkpoint**: `ckpt/step-5000`" in md
        assert "**Seed**: 42" in md
        assert "| mmlu-pro | 27.00% | 200 |" in md
        assert "| gsm8k | 5.00% | 200 |" in md
        # Failed benchmark row should show — for accuracy and the error
        assert "| bbh | — | 0 | FAILED: OOM |" in md

    def test_to_markdown_renders_per_subgroup_breakdown(self):
        card = Scorecard(
            model_checkpoint="x", model_name="y",
            eval_timestamp_utc="z",
            sample_size_per_benchmark=None,
            seed=0,
            scores=[
                BenchmarkScore(benchmark="mmlu-prox",
                              accuracy=0.30, n_total=1400,
                              per_subgroup={"en": 0.35, "hi": 0.25}),
            ],
        )
        md = card.to_markdown()
        assert "en=35.0%" in md and "hi=25.0%" in md


# --------------------------------------------------------------------------- #
# build_scorecard end-to-end
# --------------------------------------------------------------------------- #
class TestBuildScorecard:
    def test_runs_all_benchmarks_and_aggregates(self):
        bench = _MockMCQ(target_letters=["A", "B", "A", "C", "A"])
        # predict always "A" → 3 correct out of 5 → 60% accuracy
        card = build_scorecard(
            model_checkpoint="ckpt/0",
            model_name="test-model",
            benchmarks=[bench],
            predict_fn=lambda prompt: "A",
        )
        assert len(card.scores) == 1
        s = card.scores[0]
        assert s.benchmark == "mock-mcq"
        assert s.accuracy == 0.6  # 3/5
        assert s.n_total == 5
        assert s.error is None

    def test_per_subgroup_breakdown_propagates(self):
        bench = _MockMCQ(
            target_letters=["A", "A", "B", "B"],
            languages=["en", "hi", "en", "hi"],
        )
        # predict "A" → en gets 1/2, hi gets 1/2 → 50% overall
        card = build_scorecard(
            model_checkpoint="x", model_name="y",
            benchmarks=[bench], predict_fn=lambda p: "A",
        )
        s = card.scores[0]
        assert s.accuracy == 0.5
        assert s.per_subgroup["en"] == 0.5
        assert s.per_subgroup["hi"] == 0.5

    def test_failing_benchmark_recorded_but_others_run(self):
        good = _MockMCQ(["A", "A"])
        bad = _MockExploding()
        also_good = _MockMCQ(["B", "B"])
        card = build_scorecard(
            model_checkpoint="x", model_name="y",
            benchmarks=[good, bad, also_good],
            predict_fn=lambda p: "A",
        )
        # 3 results, 1 error, 2 good
        assert len(card.scores) == 3
        assert card.scores[0].error is None and card.scores[0].accuracy == 1.0
        assert card.scores[1].error is not None and "HF rate-limited" in card.scores[1].error
        assert card.scores[2].error is None and card.scores[2].accuracy == 0.0  # we predict A, target is B

    def test_sample_size_passed_through(self):
        # Create a bench with 10 examples but ask for 3 via sample_size.
        # The mock's load_examples doesn't honor sample_size (returns all),
        # so this test really just checks build_scorecard doesn't crash
        # when sample_size is given.
        bench = _MockMCQ(["A"] * 10)
        card = build_scorecard(
            model_checkpoint="x", model_name="y",
            benchmarks=[bench],
            predict_fn=lambda p: "A",
            sample_size_per_benchmark=3,
        )
        assert card.sample_size_per_benchmark == 3


# --------------------------------------------------------------------------- #
# write_scorecard — file output
# --------------------------------------------------------------------------- #
class TestWriteScorecard:
    def test_writes_json_and_md(self, tmp_path):
        card = Scorecard(
            model_checkpoint="ckpt/x",
            model_name="t",
            eval_timestamp_utc="2026-05-13T00:00:00Z",
            sample_size_per_benchmark=10,
            seed=0,
            scores=[BenchmarkScore(benchmark="b", accuracy=0.5, n_total=10)],
        )
        j, m = write_scorecard(card, tmp_path)
        assert j.exists() and m.exists()
        data = json.loads(j.read_text())
        assert data["model_checkpoint"] == "ckpt/x"
        md = m.read_text()
        assert "Release Scorecard" in md and "| b |" in md

    def test_custom_name_prefix(self, tmp_path):
        card = Scorecard(
            model_checkpoint="x", model_name="y",
            eval_timestamp_utc="z",
            sample_size_per_benchmark=None, seed=0,
        )
        j, m = write_scorecard(card, tmp_path, name_prefix="pilot_50k")
        assert j.name == "pilot_50k.json"
        assert m.name == "pilot_50k.md"
