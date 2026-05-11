"""Tests for the MILU benchmark adapter.

Validates the AI4Bharat-MILU quirk: ``answer`` is a *content string*,
not a letter. The adapter maps it back to A-D at example-load time.
Rows whose answer doesn't match any of the four options are skipped
(rather than silently mis-scored).
"""
from __future__ import annotations

import pytest

from myllm.eval.benchmarks.milu import MILUBenchmark
from myllm.eval.runner import run_benchmark
from myllm.eval.types import EvalExample


def _make_row(question, options, correct_value, domain="general"):
    return {
        "question": question,
        "option1": options[0],
        "option2": options[1],
        "option3": options[2],
        "option4": options[3],
        "answer": correct_value,
        "domain": domain,
        "language": "hindi",
    }


def _fake_provider(rows_per_lang):
    def provider(language, split):
        for r in rows_per_lang.get(language, []):
            yield r
    return provider


class TestAnswerStringMapping:
    def test_content_string_maps_to_correct_letter(self):
        rows = [_make_row("q", ["alpha", "beta", "gamma", "delta"], "gamma")]
        bench = MILUBenchmark(
            languages=("hindi",),
            n_shots=0,
            examples_provider=_fake_provider({"hindi": rows}),
        )
        ex = next(iter(bench.load_examples()))
        assert ex.target_answer == "C"

    def test_each_position_maps_correctly(self):
        for idx, letter in enumerate("ABCD"):
            opts = ["a", "b", "c", "d"]
            rows = [_make_row("q", opts, opts[idx])]
            bench = MILUBenchmark(
                languages=("hindi",),
                n_shots=0,
                examples_provider=_fake_provider({"hindi": rows}),
            )
            ex = next(iter(bench.load_examples()))
            assert ex.target_answer == letter, f"idx={idx} expected {letter}, got {ex.target_answer}"

    def test_whitespace_normalized_when_matching_answer(self):
        rows = [_make_row("q", ["alpha", "  beta  ", "gamma", "delta"], "beta")]
        bench = MILUBenchmark(
            languages=("hindi",),
            n_shots=0,
            examples_provider=_fake_provider({"hindi": rows}),
        )
        ex = next(iter(bench.load_examples()))
        assert ex.target_answer == "B"

    def test_unmatchable_answer_row_skipped(self):
        # Answer "epsilon" doesn't match any option → row dropped.
        bad = _make_row("q", ["a", "b", "c", "d"], "epsilon")
        good = _make_row("q", ["alpha", "beta", "gamma", "delta"], "beta")
        bench = MILUBenchmark(
            languages=("hindi",),
            n_shots=0,
            examples_provider=_fake_provider({"hindi": [bad, good]}),
        )
        examples = list(bench.load_examples())
        assert len(examples) == 1
        assert examples[0].target_answer == "B"


class TestPromptFormatting:
    def test_4shot_prompt_structure(self):
        rows = [
            _make_row(f"q{i}", ["o-A", "o-B", "o-C", "o-D"], "o-A")
            for i in range(6)
        ]
        bench = MILUBenchmark(
            languages=("hindi",),
            n_shots=4,
            examples_provider=_fake_provider({"hindi": rows}),
        )
        examples = list(bench.load_examples())
        # 6 rows - 4 shots = 2 test examples.
        assert len(examples) == 2
        ex = examples[0]
        # 4 shot answers + 1 open target answer = 5 "Answer:" occurrences
        # (4 ending in a letter, 1 open).
        assert ex.prompt.count("Answer:") == 5
        # Each "A. o-A" choice line present.
        assert "A. o-A" in ex.prompt
        assert "D. o-D" in ex.prompt
        # Last line is the open answer prompt.
        assert ex.prompt.rstrip().endswith("Answer:")

    def test_metadata_carries_language_and_domain(self):
        rows = [_make_row("q", ["a", "b", "c", "d"], "b", domain="physics")]
        bench = MILUBenchmark(
            languages=("hindi",),
            n_shots=0,
            examples_provider=_fake_provider({"hindi": rows}),
        )
        ex = next(iter(bench.load_examples()))
        assert ex.metadata == {"language": "hindi", "domain": "physics"}


class TestScoring:
    def setup_method(self):
        self.bench = MILUBenchmark(languages=("hindi",), examples_provider=_fake_provider({"hindi": []}))
        self.ex = EvalExample(prompt="…Answer:", target_answer="C", metadata={"language": "hindi"})

    def test_correct_letter(self):
        assert self.bench.score("C", self.ex) is True
        assert self.bench.score("c", self.ex) is True

    def test_wrong_letter(self):
        assert self.bench.score("A", self.ex) is False
        assert self.bench.score("B", self.ex) is False

    def test_letter_after_prefix(self):
        assert self.bench.score("Answer: C", self.ex) is True
        assert self.bench.score("The correct answer is C", self.ex) is True

    def test_invalid_letter(self):
        # MILU has only A-D; E onwards is invalid.
        assert self.bench.score("E", self.ex) is False
        assert self.bench.score("Z", self.ex) is False

    def test_empty(self):
        assert self.bench.score("", self.ex) is False

    def test_subgroup_returns_language(self):
        assert self.bench.subgroup_key(self.ex) == "hindi"


class TestEndToEnd:
    def test_oracle_predictor_100pct(self):
        rows = [
            _make_row("q1", ["a", "b", "c", "d"], "a"),  # → A
            _make_row("q2", ["a", "b", "c", "d"], "c"),  # → C
            _make_row("q3", ["a", "b", "c", "d"], "d"),  # → D
            _make_row("q4", ["a", "b", "c", "d"], "b"),  # → B
            _make_row("q5", ["a", "b", "c", "d"], "a"),  # → A
        ]
        bench = MILUBenchmark(
            languages=("hindi",),
            n_shots=0,
            examples_provider=_fake_provider({"hindi": rows}),
        )
        examples = list(bench.load_examples())
        targets = [ex.target_answer for ex in examples]
        i = {"v": 0}

        def predict(_p):
            ans = targets[i["v"]]
            i["v"] += 1
            return ans

        result = run_benchmark(bench, predict)
        assert result.accuracy == 1.0
        assert result.benchmark == "milu"

    def test_multilingual(self):
        rows = {
            "hindi":   [_make_row(f"प{i}", ["क", "ख", "ग", "घ"], "क") for i in range(5)],
            "bengali": [_make_row(f"প{i}", ["ক", "খ", "গ", "ঘ"], "গ") for i in range(5)],
        }
        bench = MILUBenchmark(
            languages=("hindi", "bengali"),
            n_shots=0,
            examples_provider=_fake_provider(rows),
        )
        examples = list(bench.load_examples())
        langs = {ex.metadata["language"] for ex in examples}
        assert langs == {"hindi", "bengali"}
        assert len(examples) == 10
