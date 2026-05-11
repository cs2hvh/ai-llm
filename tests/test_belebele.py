"""Tests for the Belebele benchmark adapter.

Uses ``examples_provider`` to bypass HF and supply synthetic RC rows.
Verifies:

  - Prompt formatting: passage + question + 4 choices + "Answer:" cue.
  - Gold-answer conversion: 1-indexed correct_answer_num → letter A-D.
  - Answer extraction: same heuristic as MMLU-ProX.
  - Subgroup keying: per-language stratification (FLORES BCP-47 codes).
  - Multi-language iteration coverage.
"""
from __future__ import annotations

import pytest

from myllm.eval.benchmarks.belebele import BelebeleBenchmark
from myllm.eval.runner import run_benchmark
from myllm.eval.types import EvalExample


def _make_row(passage, question, choices, correct_1indexed):
    return {
        "flores_passage": passage,
        "question": question,
        "mc_answer1": choices[0],
        "mc_answer2": choices[1],
        "mc_answer3": choices[2],
        "mc_answer4": choices[3],
        "correct_answer_num": str(correct_1indexed),
    }


def _fake_provider(rows_per_lang):
    def provider(language, split):
        for r in rows_per_lang.get(language, []):
            yield r
    return provider


class TestPromptFormatting:
    def test_prompt_structure(self):
        rows = [_make_row("A passage about cats.", "Are cats furry?", ["yes", "no", "sometimes", "always"], 1)]
        bench = BelebeleBenchmark(languages=("eng_Latn",), examples_provider=_fake_provider({"eng_Latn": rows}))
        examples = list(bench.load_examples())
        assert len(examples) == 1
        ex = examples[0]
        assert "Passage: A passage about cats." in ex.prompt
        assert "Question: Are cats furry?" in ex.prompt
        assert "A. yes" in ex.prompt
        assert "B. no" in ex.prompt
        assert "C. sometimes" in ex.prompt
        assert "D. always" in ex.prompt
        assert ex.prompt.rstrip().endswith("Answer:")
        # 1-indexed gold=1 → letter A.
        assert ex.target_answer == "A"

    def test_gold_letter_mapping(self):
        for gold_1idx, gold_letter in [(1, "A"), (2, "B"), (3, "C"), (4, "D")]:
            row = _make_row("p", "q", ["o1", "o2", "o3", "o4"], gold_1idx)
            bench = BelebeleBenchmark(languages=("eng_Latn",), examples_provider=_fake_provider({"eng_Latn": [row]}))
            ex = next(iter(bench.load_examples()))
            assert ex.target_answer == gold_letter

    def test_language_in_metadata(self):
        rows = [_make_row("p", "q", ["a", "b", "c", "d"], 2)]
        bench = BelebeleBenchmark(languages=("hin_Deva",), examples_provider=_fake_provider({"hin_Deva": rows}))
        ex = next(iter(bench.load_examples()))
        assert ex.metadata["language"] == "hin_Deva"

    def test_malformed_row_skipped_with_warning(self):
        """Row missing a required field is skipped, not crashes."""
        bad_row = {"flores_passage": "p"}  # missing question, choices, gold
        good_row = _make_row("p", "q", ["a", "b", "c", "d"], 1)
        bench = BelebeleBenchmark(
            languages=("eng_Latn",),
            examples_provider=_fake_provider({"eng_Latn": [bad_row, good_row]}),
        )
        examples = list(bench.load_examples())
        # Only the good row survived.
        assert len(examples) == 1


class TestAnswerScoring:
    def setup_method(self):
        rows = [_make_row("p", "q", ["a", "b", "c", "d"], 2)]
        self.bench = BelebeleBenchmark(
            languages=("eng_Latn",),
            examples_provider=_fake_provider({"eng_Latn": rows}),
        )
        self.ex = EvalExample(prompt="…Answer:", target_answer="B", metadata={"language": "eng_Latn"})

    def test_letter_only(self):
        assert self.bench.score("B", self.ex) is True
        assert self.bench.score("A", self.ex) is False

    def test_letter_with_prefix(self):
        assert self.bench.score("The answer is B.", self.ex) is True
        assert self.bench.score("Answer: B", self.ex) is True
        assert self.bench.score("(B)", self.ex) is True

    def test_letter_E_or_beyond_is_invalid(self):
        """Belebele only has A B C D — letters beyond that are not valid answers."""
        ex_with_E = EvalExample(prompt="p", target_answer="A", metadata={"language": "eng_Latn"})
        # Model says "E" — not a valid Belebele answer, so score=False.
        assert self.bench.score("E", ex_with_E) is False
        # Same for F, ..., J that would be valid in MMLU-ProX:
        assert self.bench.score("F", ex_with_E) is False
        assert self.bench.score("J", ex_with_E) is False

    def test_empty_response(self):
        assert self.bench.score("", self.ex) is False

    def test_subgroup_key_returns_language(self):
        assert self.bench.subgroup_key(self.ex) == "eng_Latn"


class TestMultilingualIteration:
    def test_multiple_languages_yield_all_examples(self):
        provider = _fake_provider({
            "eng_Latn": [_make_row(f"p{i}", "q", ["a", "b", "c", "d"], 1) for i in range(3)],
            "hin_Deva": [_make_row(f"प{i}", "?", ["क", "ख", "ग", "घ"], 2) for i in range(2)],
            "arb_Arab": [_make_row(f"م{i}", "؟", ["ا", "ب", "ج", "د"], 3) for i in range(4)],
        })
        bench = BelebeleBenchmark(
            languages=("eng_Latn", "hin_Deva", "arb_Arab"),
            examples_provider=provider,
        )
        examples = list(bench.load_examples())
        assert len(examples) == 9
        by_lang = {}
        for ex in examples:
            by_lang.setdefault(ex.metadata["language"], 0)
            by_lang[ex.metadata["language"]] += 1
        assert by_lang == {"eng_Latn": 3, "hin_Deva": 2, "arb_Arab": 4}

    def test_sample_size_caps_per_language(self):
        provider = _fake_provider({
            "eng_Latn": [_make_row(f"p{i}", "q", ["a", "b", "c", "d"], 1) for i in range(10)],
            "hin_Deva": [_make_row(f"प{i}", "?", ["क", "ख", "ग", "घ"], 2) for i in range(10)],
        })
        bench = BelebeleBenchmark(
            languages=("eng_Latn", "hin_Deva"),
            examples_provider=provider,
        )
        examples = list(bench.load_examples(sample_size=3))
        # 3 per language × 2 languages = 6.
        assert len(examples) == 6


class TestEndToEndWithRunner:
    def test_oracle_predictor_scores_100pct(self):
        rows = [
            _make_row("p1", "q", ["a", "b", "c", "d"], 1),  # gold = A
            _make_row("p2", "q", ["a", "b", "c", "d"], 3),  # gold = C
            _make_row("p3", "q", ["a", "b", "c", "d"], 4),  # gold = D
        ]
        bench = BelebeleBenchmark(
            languages=("eng_Latn",),
            examples_provider=_fake_provider({"eng_Latn": rows}),
        )

        # Cheating oracle: read the gold letter directly from the prompt's
        # Question line (we encoded gold via the rows themselves; the
        # cleanest test is to use load_examples to get targets, then
        # replay them through a predictor that returns them in order).
        examples = list(bench.load_examples())
        targets = [ex.target_answer for ex in examples]
        i = {"v": 0}

        def predict(_prompt):
            ans = targets[i["v"]]
            i["v"] += 1
            return ans

        result = run_benchmark(bench, predict)
        assert result.accuracy == 1.0
        assert result.n_correct == 3 and result.n_total == 3

    def test_naive_always_a_predictor(self):
        rows = [
            _make_row("p", "q", ["a", "b", "c", "d"], 1),  # gold = A
            _make_row("p", "q", ["a", "b", "c", "d"], 2),  # gold = B
            _make_row("p", "q", ["a", "b", "c", "d"], 3),  # gold = C
            _make_row("p", "q", ["a", "b", "c", "d"], 4),  # gold = D
        ]
        bench = BelebeleBenchmark(
            languages=("eng_Latn",),
            examples_provider=_fake_provider({"eng_Latn": rows}),
        )
        result = run_benchmark(bench, lambda _: "A")
        # Only the first row's gold is A → 1/4 = 25% accuracy.
        assert result.accuracy == 0.25
