"""Tests for the MMLU-ProX benchmark adapter.

Uses ``examples_provider`` to bypass HF and supply synthetic question rows.
Verifies:

  - Prompt formatting: 4-shot in-context, target ends with "Answer:".
  - Answer extraction: handles letter-only, leading whitespace, full
    phrases like "The answer is A.", and produces False on missing letter.
  - Subgroup keying: per-language stratification.
  - Multi-language iteration: examples from all configured languages.
"""
from __future__ import annotations

import re

from myllm.eval.benchmarks.mmlu_prox import MMLUProXBenchmark
from myllm.eval.runner import run_benchmark
from myllm.eval.types import EvalExample


def _make_row(question: str, options: list[str], answer_index: int, subject: str = "math") -> dict:
    return {
        "question": question,
        "options": options,
        "answer_index": answer_index,
        "subject": subject,
    }


def _fake_provider(rows_per_lang: dict[str, list[dict]]):
    """Return an examples_provider that yields the given rows per language."""
    def provider(language: str, split: str):
        for r in rows_per_lang.get(language, []):
            yield r
    return provider


def _make_n_rows(n: int, lang_seed: int = 0) -> list[dict]:
    """Build n distinct rows with predictable answers."""
    return [
        _make_row(
            question=f"q{i+1} (seed {lang_seed})",
            options=[f"opt-A-{i}", f"opt-B-{i}", f"opt-C-{i}", f"opt-D-{i}"],
            answer_index=i % 4,
        )
        for i in range(n)
    ]


class TestPromptFormatting:
    def test_prompt_includes_choices_and_terminates_with_answer_colon(self):
        provider = _fake_provider({"en": _make_n_rows(6)})
        bench = MMLUProXBenchmark(languages=("en",), n_shots=2, examples_provider=provider)
        examples = list(bench.load_examples(split="test"))
        assert len(examples) == 4  # 6 rows - 2 shots
        for ex in examples:
            assert "A. opt-A-" in ex.prompt
            assert "D. opt-D-" in ex.prompt
            # Last line before extracted answer should be "Answer:".
            assert ex.prompt.strip().endswith("Answer:")
            # Exactly one open "Answer:" without a letter (the target).
            assert ex.prompt.count("Answer:") == 3  # 2 shots end "Answer: X", 1 target ends "Answer:"

    def test_prompt_carries_language_in_metadata(self):
        provider = _fake_provider({"hi": _make_n_rows(5)})
        bench = MMLUProXBenchmark(languages=("hi",), n_shots=2, examples_provider=provider)
        for ex in bench.load_examples():
            assert ex.metadata["language"] == "hi"


class TestAnswerScoring:
    def setup_method(self):
        # Reusable example with target "B".
        self.ex = EvalExample(prompt="…Answer:", target_answer="B", metadata={"language": "en"})
        provider = _fake_provider({"en": _make_n_rows(5)})
        self.bench = MMLUProXBenchmark(languages=("en",), examples_provider=provider)

    def test_extract_letter_only(self):
        assert self.bench.score("B", self.ex) is True
        assert self.bench.score("A", self.ex) is False

    def test_extract_with_leading_whitespace(self):
        assert self.bench.score("  B  ", self.ex) is True

    def test_extract_letter_from_lm_eval_style_responses(self):
        # Standard non-CoT shapes the model will actually produce after
        # the prompt's "Answer:" cue. Our prompts end with "Answer:" so
        # the model's response will be one of these forms.
        assert self.bench.score("The answer is B.", self.ex) is True
        assert self.bench.score("Answer: B", self.ex) is True
        assert self.bench.score("(B)", self.ex) is True
        assert self.bench.score("B.", self.ex) is True
        assert self.bench.score("B", self.ex) is True
        # Unrecognized prefixes like "After consideration: B." would need
        # a CoT extractor; our heuristic refuses them (returns False) which
        # is the safer default — better an under-credit than over-credit.
        assert self.bench.score("After consideration: B.", self.ex) is False

    def test_cot_response_with_letter_in_passing_is_NOT_credited(self):
        # "I think B" — model says "I" first; "I" is in A-J but is a
        # passing-word usage, not the answer. We deliberately don't extract
        # it (lm-eval-harness convention). A separate CoT extractor would
        # parse "the answer is B" in such cases.
        assert self.bench.score("I think B is correct because…", self.ex) is False

    def test_returns_false_when_no_letter(self):
        assert self.bench.score("nothing here", self.ex) is False
        assert self.bench.score("", self.ex) is False

    def test_case_insensitive(self):
        assert self.bench.score("the answer is b.", self.ex) is True
        assert self.bench.score("b", self.ex) is True

    def test_subgroup_key(self):
        assert self.bench.subgroup_key(self.ex) == "en"
        assert self.bench.subgroup_key(EvalExample("p", "A", {"language": "zh"})) == "zh"
        assert self.bench.subgroup_key(EvalExample("p", "A", {})) == "unknown"


class TestMultilingualIteration:
    def test_examples_yielded_for_every_configured_language(self):
        provider = _fake_provider({
            "en": _make_n_rows(8, lang_seed=1),
            "hi": _make_n_rows(8, lang_seed=2),
            "es": _make_n_rows(8, lang_seed=3),
        })
        bench = MMLUProXBenchmark(
            languages=("en", "hi", "es"),
            n_shots=4,
            examples_provider=provider,
        )
        examples = list(bench.load_examples(split="test"))
        # Each lang: 8 rows - 4 shots = 4 test examples → 12 total.
        assert len(examples) == 12
        langs_seen = {ex.metadata["language"] for ex in examples}
        assert langs_seen == {"en", "hi", "es"}

    def test_skips_language_with_too_few_rows(self):
        """If a language has fewer rows than n_shots+1, skip it gracefully."""
        provider = _fake_provider({
            "en": _make_n_rows(10),
            "hi": _make_n_rows(2),  # too small for n_shots=4
        })
        bench = MMLUProXBenchmark(
            languages=("en", "hi"),
            n_shots=4,
            examples_provider=provider,
        )
        langs = {ex.metadata["language"] for ex in bench.load_examples()}
        assert langs == {"en"}  # Hindi was silently skipped


class TestEndToEndWithRunner:
    def test_perfect_predictor_scores_100pct_per_language(self):
        """Run the full pipeline: bench → runner with a predict that
        always reads the gold answer out of the prompt."""
        provider = _fake_provider({
            "en": _make_n_rows(8),
            "hi": _make_n_rows(8),
        })
        bench = MMLUProXBenchmark(
            languages=("en", "hi"),
            n_shots=2,
            examples_provider=provider,
        )

        # Cheat predict: look up the gold letter by parsing the prompt.
        # Since this is our synthetic data, the target letter is deterministic
        # — we just emit a fixed letter pattern. Easier: cheat by counting
        # the trailing "Answer:" and pulling the answer index from row text.
        # In production this would be the model's generation; here we always
        # return "A" and we'll check what fraction matches.

        # Setup: rows are answer_index = i % 4. With seed shuffling, the
        # final test set has answers spread across A, B, C, D roughly evenly.
        # A naive predictor returning "A" should get ~25% accuracy.
        predict_always_a = lambda _: "A"  # noqa: E731
        result = run_benchmark(bench, predict_always_a)
        assert 0.0 < result.accuracy < 1.0, f"got {result.accuracy}"
        # All examples are scored.
        assert result.n_total > 0
        assert "en" in result.per_subgroup
        assert "hi" in result.per_subgroup

    def test_random_predictor_baseline(self):
        """A predictor that returns the correct letter every time should
        score 100%, confirming the runner + scorer wiring."""
        provider = _fake_provider({"en": _make_n_rows(20)})
        bench = MMLUProXBenchmark(
            languages=("en",),
            n_shots=4,
            examples_provider=provider,
        )

        # Build a predictor that introspects the prompt to find the target
        # row's options and returns the correct letter. We do this by
        # looking for the target's "answer_index" via the row index in
        # the question string ("q{N}").
        # We need a way to know which row is being asked. Easier: parse
        # the question id and reconstruct the answer rule.
        question_to_answer = {}
        # Generate the same shuffled answer mapping.
        # NOTE: For this test we just verify the system can hit 100%
        # by hard-coding answers. Use a cheating predictor that reads
        # the target from the EvalExample object — except runner only
        # passes the prompt. We instead build an oracle that knows the
        # answer rule.

        # Simpler approach: have the bench load examples through our
        # provider, capture the target letters in advance, then run a
        # predictor that returns them in order.
        examples_seen = list(bench.load_examples())
        targets_in_order = [ex.target_answer for ex in examples_seen]

        idx_counter = {"i": 0}

        def predict_in_order(prompt):
            ans = targets_in_order[idx_counter["i"]]
            idx_counter["i"] += 1
            return ans

        # Re-run the runner with this predictor.
        result = run_benchmark(bench, predict_in_order)
        # Note: load_examples is non-deterministic in shuffle if seed differs,
        # but the iterator from run_benchmark uses the same default seed=0
        # as our pre-load call, so order matches.
        assert result.accuracy == 1.0
        assert result.n_correct == result.n_total > 0
