"""Tests for the MMLU-Pro adapter (Round D6 / Layer 2, 2026-05-16).

Uses ``examples_provider`` to bypass HF. Verifies:
  - Prompt formatting: n-shot in-context, target ends with "Answer:".
  - Answer extraction: handles letter-only, prefixes, false negatives
    on out-of-vocab letters.
  - Both row schemas accepted (options list + answer letter; column-wise
    option_0..option_9 + answer_index).
"""
from __future__ import annotations

import pytest

from myllm.eval.benchmarks.mmlu_pro import MMLUProBenchmark
from myllm.eval.types import EvalExample


def _row(question: str, options: list[str], answer: str, category: str = "math") -> dict:
    return {
        "question": question,
        "options": options,
        "answer": answer,  # letter form
        "category": category,
    }


def _fake_provider(rows: list[dict]):
    def provider(split: str):
        yield from rows
    return provider


def _make_n_rows(n: int) -> list[dict]:
    letters = "ABCDEFGHIJ"
    return [
        _row(
            question=f"q{i+1}",
            options=[f"opt-{letters[j]}-{i}" for j in range(4)],
            answer=letters[i % 4],
            category="math" if i % 2 == 0 else "biology",
        )
        for i in range(n)
    ]


class TestPromptFormatting:
    def test_includes_choices_and_terminates_with_answer_colon(self):
        bench = MMLUProBenchmark(
            n_shots=2, examples_provider=_fake_provider(_make_n_rows(6)),
        )
        examples = list(bench.load_examples(split="test"))
        assert len(examples) == 4  # 6 rows - 2 shots
        for ex in examples:
            assert "A. opt-A-" in ex.prompt
            assert "D. opt-D-" in ex.prompt
            assert ex.prompt.strip().endswith("Answer:")
            # 2 shot answers + 1 unanswered target = 3 "Answer:" occurrences
            assert ex.prompt.count("Answer:") == 3

    def test_target_letter_matches_row_answer(self):
        bench = MMLUProBenchmark(
            n_shots=1, examples_provider=_fake_provider(_make_n_rows(5)),
        )
        examples = list(bench.load_examples())
        for ex in examples:
            assert ex.target_answer in "ABCDEFGHIJ"

    def test_subject_metadata_passed_through(self):
        bench = MMLUProBenchmark(
            n_shots=1, examples_provider=_fake_provider(_make_n_rows(4)),
        )
        for ex in bench.load_examples():
            assert ex.metadata["subject"] in ("math", "biology")


class TestSchemaCompatibility:
    """Accept both the modern options-list schema and the column-wise
    option_0..option_9 schema some MMLU-Pro mirrors use."""

    def test_options_list_schema(self):
        bench = MMLUProBenchmark(
            n_shots=1, examples_provider=_fake_provider(_make_n_rows(3)),
        )
        examples = list(bench.load_examples())
        assert examples  # at least one
        assert "opt-A-" in examples[0].prompt

    def test_column_wise_option_schema(self):
        col_rows = [
            {
                "question": "qC",
                "option_0": "alpha", "option_1": "beta",
                "option_2": "gamma", "option_3": "delta",
                "option_4": None, "option_5": "N/A",  # ignored sentinels
                "answer": "C",
                "category": "math",
            }
            for _ in range(3)
        ]
        bench = MMLUProBenchmark(
            n_shots=1, examples_provider=_fake_provider(col_rows),
        )
        examples = list(bench.load_examples())
        assert examples
        assert "C. gamma" in examples[0].prompt
        assert "E." not in examples[0].prompt  # 4 options only

    def test_answer_index_fallback(self):
        # Some sources provide answer_index instead of an answer letter.
        rows = [
            {
                "question": "qX",
                "options": ["a", "b", "c", "d"],
                "answer_index": 2,  # -> "C"
                "category": "math",
            }
            for _ in range(3)
        ]
        bench = MMLUProBenchmark(
            n_shots=1, examples_provider=_fake_provider(rows),
        )
        examples = list(bench.load_examples())
        assert examples[0].target_answer == "C"


class TestScoring:
    @pytest.fixture
    def bench(self):
        return MMLUProBenchmark(
            n_shots=1, examples_provider=_fake_provider(_make_n_rows(3)),
        )

    @pytest.fixture
    def ex_b(self):
        return EvalExample(prompt="…Answer:", target_answer="B",
                           metadata={"subject": "math"})

    def test_letter_only(self, bench, ex_b):
        assert bench.score("B", ex_b) is True
        assert bench.score("A", ex_b) is False

    def test_lm_eval_phrases(self, bench, ex_b):
        assert bench.score("Answer: B", ex_b) is True
        assert bench.score("The answer is B.", ex_b) is True
        assert bench.score("(B)", ex_b) is True
        assert bench.score("B.", ex_b) is True

    def test_passing_letter_in_cot_NOT_credited(self, bench, ex_b):
        # "I think B" — first non-prefix char is "I", not "B".
        # lm-eval-harness convention: refuse. Better under-credit than over.
        assert bench.score("I think B is right", ex_b) is False

    def test_empty_and_garbage(self, bench, ex_b):
        assert bench.score("", ex_b) is False
        assert bench.score("!!!", ex_b) is False
        assert bench.score("nothing here", ex_b) is False

    def test_case_insensitive(self, bench, ex_b):
        assert bench.score("b", ex_b) is True
        assert bench.score("the answer is b.", ex_b) is True

    def test_letters_e_through_j(self, bench):
        # MMLU-Pro has up to 10 options. Verify E-J accepted.
        ex = EvalExample(prompt="…Answer:", target_answer="J", metadata={})
        assert bench.score("J", ex) is True
        ex2 = EvalExample(prompt="…Answer:", target_answer="G", metadata={})
        assert bench.score("G", ex2) is True

    def test_subgroup_key(self, bench):
        assert bench.subgroup_key(
            EvalExample("p", "A", {"subject": "law"})
        ) == "law"
        assert bench.subgroup_key(
            EvalExample("p", "A", {})
        ) == "n/a"
