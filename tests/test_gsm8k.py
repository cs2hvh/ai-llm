"""Tests for the GSM8K adapter (Round D6 / Layer 2, 2026-05-16).

Uses ``examples_provider`` to bypass HF. Verifies:
  - Prompt formatting: n-shot CoT, target ends with "A:".
  - Gold answer extraction from "#### N" markers.
  - Score: prefers `#### N` over fallback last-number; normalizes
    "5.0" == "5"; refuses empty / number-less predictions.
"""
from __future__ import annotations

import pytest

from myllm.eval.benchmarks.gsm8k import (
    GSM8KBenchmark,
    _extract_predicted_number,
    _gold_answer_from_row,
    _normalize_number,
)
from myllm.eval.types import EvalExample


def _row(q: str, cot: str, final: int) -> dict:
    return {
        "question": q,
        "answer": f"{cot}\n#### {final}",
    }


def _fake_provider(rows_per_split: dict[str, list[dict]]):
    def provider(split: str):
        yield from rows_per_split.get(split, [])
    return provider


def _train_and_test_rows(n_train: int, n_test: int) -> dict[str, list[dict]]:
    return {
        "train": [_row(f"trainq{i}", f"reasoning {i}", i) for i in range(n_train)],
        "test":  [_row(f"testq{i}",  f"work {i}",       100 + i) for i in range(n_test)],
    }


class TestGoldAnswerExtraction:
    def test_simple_integer(self):
        assert _gold_answer_from_row({"answer": "Step 1. Step 2.\n#### 42"}) == "42"

    def test_negative_integer(self):
        assert _gold_answer_from_row({"answer": "blah\n#### -7"}) == "-7"

    def test_no_marker_raises(self):
        with pytest.raises(ValueError, match="missing '#### N'"):
            _gold_answer_from_row({"answer": "no marker here"})

    def test_uses_first_marker(self):
        # Real GSM8K answers have exactly one marker; this test pins
        # that the regex grabs the first hit if there happen to be two.
        out = _gold_answer_from_row({"answer": "interim\n#### 5\nmore\n#### 99"})
        assert out == "5"


class TestNumberNormalization:
    def test_int_string_unchanged(self):
        assert _normalize_number("5") == "5"
        assert _normalize_number("-7") == "-7"

    def test_float_becomes_int_when_integer_valued(self):
        assert _normalize_number("5.0") == "5"
        assert _normalize_number("-7.00") == "-7"

    def test_true_float_preserved(self):
        assert _normalize_number("5.5") == "5.5"

    def test_garbage_passes_through(self):
        assert _normalize_number("abc") == "abc"


class TestPredictionExtraction:
    def test_marker_priority(self):
        # When the model emits CoT + #### marker, that's the answer.
        pred = "Let me work this out. 3 + 2 = 5. #### 5"
        assert _extract_predicted_number(pred) == "5"

    def test_last_marker_wins(self):
        # If the model emits multiple markers (unusual), take the last —
        # represents the model's "final" answer after corrections.
        pred = "First attempt: #### 4. Wait actually #### 5"
        assert _extract_predicted_number(pred) == "5"

    def test_fallback_to_last_number(self):
        # Model didn't produce the canonical marker.
        assert _extract_predicted_number("the answer is 42") == "42"

    def test_no_number_returns_none(self):
        assert _extract_predicted_number("I don't know") is None
        assert _extract_predicted_number("") is None

    def test_normalizes_float(self):
        assert _extract_predicted_number("#### 5.0") == "5"


class TestPromptFormatting:
    def test_shots_drawn_from_train_split(self):
        bench = GSM8KBenchmark(
            n_shots=2,
            examples_provider=_fake_provider(_train_and_test_rows(5, 3)),
        )
        examples = list(bench.load_examples(split="test"))
        assert len(examples) == 3
        # Test prompts must mention train questions (the shots).
        joined = examples[0].prompt
        assert "trainq" in joined
        assert "testq" in joined
        # Last line is the target unanswered "A:".
        assert joined.strip().endswith("A:")

    def test_target_is_normalized_integer_string(self):
        bench = GSM8KBenchmark(
            n_shots=1,
            examples_provider=_fake_provider(_train_and_test_rows(2, 2)),
        )
        for ex in bench.load_examples():
            # Test rows had final=100,101. Both are valid normalized ints.
            assert ex.target_answer in ("100", "101")

    def test_fallback_when_train_split_missing(self):
        # Provider only has test split. The adapter should still work
        # using the head of the test split as shots (with a warning).
        bench = GSM8KBenchmark(
            n_shots=1,
            examples_provider=_fake_provider({"test": _train_and_test_rows(0, 5)["test"]}),
        )
        examples = list(bench.load_examples())
        # 5 test rows - 1 shot = 4 examples.
        assert len(examples) == 4

    def test_sample_size_caps_examples(self):
        bench = GSM8KBenchmark(
            n_shots=2,
            examples_provider=_fake_provider(_train_and_test_rows(5, 100)),
        )
        examples = list(bench.load_examples(sample_size=7))
        assert len(examples) == 7


class TestScoring:
    @pytest.fixture
    def bench(self):
        return GSM8KBenchmark(
            n_shots=1,
            examples_provider=_fake_provider(_train_and_test_rows(2, 1)),
        )

    @pytest.fixture
    def ex_42(self):
        return EvalExample(prompt="Q: blah\nA:", target_answer="42", metadata={})

    def test_correct_with_marker(self, bench, ex_42):
        assert bench.score(
            "Let me think. 6 * 7 = 42. #### 42", ex_42,
        ) is True

    def test_correct_with_fallback_last_number(self, bench, ex_42):
        # Model didn't produce the marker but ended with the right number.
        assert bench.score("the answer is 42", ex_42) is True

    def test_wrong_number_fails(self, bench, ex_42):
        assert bench.score("the answer is 41 #### 41", ex_42) is False

    def test_no_number_fails(self, bench, ex_42):
        assert bench.score("I don't know", ex_42) is False
        assert bench.score("", ex_42) is False

    def test_float_normalized_to_int_matches(self, bench, ex_42):
        # Model says "42.0", gold is "42" — should match.
        assert bench.score("#### 42.0", ex_42) is True

    def test_marker_overrides_intermediate_numbers(self, bench, ex_42):
        # CoT mentions wrong intermediate numbers but #### is right.
        assert bench.score(
            "6 + 7 = 13. Wait, 6 * 7 = 42. #### 42", ex_42,
        ) is True

    def test_subgroup_is_all(self, bench, ex_42):
        assert bench.subgroup_key(ex_42) == "all"
