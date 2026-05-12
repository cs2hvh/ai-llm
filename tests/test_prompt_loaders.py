"""Tests for the v1-gate prompt loaders used by the decontamination index.

These tests pin the *contract* between the loader and the HF row schema:
if a benchmark's HF row schema changes (or we mis-remember it), the
loader silently skips every row and yields zero prompts — that would
silently produce a half-empty decontamination index and let
contaminated docs through.

We inject a fake examples_provider so the tests run without HF.
"""
from __future__ import annotations

from collections.abc import Iterator

import pytest

from myllm.data.prompt_loaders import (
    PROMPT_LOADERS,
    iter_bbh,
    iter_gsm8k,
    iter_humaneval_plus,
    iter_ifeval,
    iter_math,
    iter_mbpp_plus,
    iter_mgsm,
    iter_mmlu_pro,
    load_prompts,
)


def _fixture_provider(rows_by_split: dict[tuple[str, str], list[dict]]):
    """Build an examples_provider keyed on (dataset_id, split).

    The provider signature in the loaders is ``(dataset_id, split) -> Iterator[dict]``;
    we ignore dataset_id in tests since it's a fixed default per loader.
    """

    def _provider(dataset_id: str, split: str) -> Iterator[dict]:
        # We key on split alone since each loader has a fixed dataset_id.
        for (_, s), rows in rows_by_split.items():
            if s == split:
                yield from rows
                return
        return

    return _provider


# --------------------------------------------------------------------------- #
# MMLU-Pro: question + options must both reach the index
# --------------------------------------------------------------------------- #
def test_mmlu_pro_yields_question_and_options():
    provider = _fixture_provider(
        {
            ("any", "test"): [
                {
                    "question": "What is the capital of France?",
                    "options": ["London", "Paris", "Berlin", "Madrid"],
                    "answer": "B",
                },
            ]
        }
    )
    prompts = list(iter_mmlu_pro(examples_provider=provider))
    assert len(prompts) == 1
    assert "capital of France" in prompts[0]
    assert "Paris" in prompts[0]  # options must be in the indexed text


def test_mmlu_pro_skips_empty_questions():
    provider = _fixture_provider(
        {
            ("any", "test"): [
                {"question": "", "options": ["a", "b"]},
                {"question": "   ", "options": ["c", "d"]},
                {"question": "Real question", "options": ["a", "b"]},
            ]
        }
    )
    prompts = list(iter_mmlu_pro(examples_provider=provider))
    assert prompts == ["Real question a b"]


def test_mmlu_pro_sample_size_caps_yield():
    provider = _fixture_provider(
        {("any", "test"): [{"question": f"q{i}", "options": []} for i in range(10)]}
    )
    prompts = list(iter_mmlu_pro(sample_size=3, examples_provider=provider))
    assert len(prompts) == 3


# --------------------------------------------------------------------------- #
# HumanEval+ / MBPP+: code-prompt extraction
# --------------------------------------------------------------------------- #
def test_humaneval_plus_yields_prompt_field():
    provider = _fixture_provider(
        {
            ("any", "test"): [
                {"task_id": "HumanEval/0", "prompt": "def add(a, b):\n    '''Sum'''\n"},
                {"task_id": "HumanEval/1", "prompt": "def sub(a, b):\n    return"},
            ]
        }
    )
    prompts = list(iter_humaneval_plus(examples_provider=provider))
    assert len(prompts) == 2
    assert "def add" in prompts[0]
    assert "def sub" in prompts[1]


def test_mbpp_plus_falls_back_from_prompt_to_text():
    """MBPP variants use either `prompt` or `text`."""
    provider = _fixture_provider(
        {
            ("any", "test"): [
                {"task_id": 1, "prompt": "Write a python function..."},
                {"task_id": 2, "text": "Implement a function..."},  # legacy field
            ]
        }
    )
    prompts = list(iter_mbpp_plus(examples_provider=provider))
    assert prompts == [
        "Write a python function...",
        "Implement a function...",
    ]


# --------------------------------------------------------------------------- #
# GSM8K / MATH: math problem text
# --------------------------------------------------------------------------- #
def test_gsm8k_yields_question_only():
    """Answer must NOT be in the indexed text — it's what we protect."""
    provider = _fixture_provider(
        {
            ("any", "test"): [
                {
                    "question": "Janet has 3 apples. She buys 5 more. How many?",
                    "answer": "8\n#### 8",
                },
            ]
        }
    )
    prompts = list(iter_gsm8k(examples_provider=provider))
    assert len(prompts) == 1
    assert "How many?" in prompts[0]
    assert "8" not in prompts[0]


def test_math_yields_problem_field():
    provider = _fixture_provider(
        {
            ("any", "test"): [
                {"problem": "Find x such that x^2 = 16.", "solution": "...", "answer": "4"},
            ]
        }
    )
    prompts = list(iter_math(examples_provider=provider))
    assert prompts == ["Find x such that x^2 = 16."]


# --------------------------------------------------------------------------- #
# MGSM: multilingual — must iterate over the configured languages
# --------------------------------------------------------------------------- #
def test_mgsm_iterates_all_languages():
    """The loader iterates each language config; each yields its rows."""
    seen_langs: list[str] = []

    def _provider(dataset_id: str, split: str) -> Iterator[dict]:
        # In real HF the language is the `name` config arg; here we
        # don't see it directly. Simulate by yielding one row per call
        # and tracking via a closure counter.
        seen_langs.append(split)  # split is constant; we just count calls
        yield {"question": f"call_{len(seen_langs)}", "answer": "1"}

    prompts = list(
        iter_mgsm(
            languages=("en", "es", "fr"),
            examples_provider=_provider,
        )
    )
    # 3 languages × 1 row each = 3 prompts
    assert len(prompts) == 3
    assert len(seen_langs) == 3


def test_mgsm_sample_size_is_per_language():
    """sample_size caps per-language yield, not total."""

    def _provider(dataset_id: str, split: str) -> Iterator[dict]:
        for i in range(10):
            yield {"question": f"q{i}", "answer": "1"}

    prompts = list(
        iter_mgsm(
            languages=("en", "es"),
            sample_size=2,
            examples_provider=_provider,
        )
    )
    # 2 langs × 2 per lang = 4 prompts
    assert len(prompts) == 4


# --------------------------------------------------------------------------- #
# BBH: multi-subtask iteration
# --------------------------------------------------------------------------- #
def test_bbh_iterates_subtasks():
    def _provider(dataset_id: str, split: str) -> Iterator[dict]:
        yield {"input": "Is the following claim true?", "target": "Yes"}

    prompts = list(
        iter_bbh(
            subtasks=("boolean_expressions", "navigate"),
            examples_provider=_provider,
        )
    )
    assert len(prompts) == 2
    assert all("Is the following claim true?" in p for p in prompts)


def test_bbh_default_subtasks_present():
    """The default subtask tuple must cover the canonical 27 BBH tasks."""
    # We test by yielding one prompt per subtask and counting.
    def _provider(dataset_id: str, split: str) -> Iterator[dict]:
        yield {"input": "x"}

    prompts = list(iter_bbh(examples_provider=_provider))
    # BBH has 27 subtasks in the canonical Suzgun et al. release
    assert len(prompts) == 27


# --------------------------------------------------------------------------- #
# IFEval
# --------------------------------------------------------------------------- #
def test_ifeval_yields_prompt():
    provider = _fixture_provider(
        {
            ("any", "train"): [
                {"key": 1, "prompt": "Write a response without using the letter e."},
                {"key": 2, "prompt": "Write exactly 3 paragraphs."},
            ]
        }
    )
    prompts = list(iter_ifeval(examples_provider=provider))
    assert len(prompts) == 2
    assert "letter e" in prompts[0]


# --------------------------------------------------------------------------- #
# Defensive: very long prompts get truncated before hashing
# --------------------------------------------------------------------------- #
def test_loader_truncates_oversized_prompts():
    """A 100KB prompt (e.g. RULER-style long-context) should be capped to
    avoid blowing up the n-gram index with millions of useless entries."""
    long_text = "abcd " * 5000  # ~25KB
    provider = _fixture_provider(
        {("any", "test"): [{"problem": long_text}]}
    )
    prompts = list(iter_math(examples_provider=provider))
    assert len(prompts) == 1
    assert len(prompts[0]) <= 8000


# --------------------------------------------------------------------------- #
# Registry / dispatch helper
# --------------------------------------------------------------------------- #
def test_registry_covers_extended_gate_set():
    """The v1 model card commits to indexing this exact set; this test
    prevents silent regressions if a loader is removed by mistake."""
    expected = {
        "mmlu-pro",
        "humaneval-plus",
        "mbpp-plus",
        "gsm8k",
        "math",
        "mgsm",
        "bbh",
        "ifeval",
    }
    assert expected.issubset(set(PROMPT_LOADERS.keys()))


def test_load_prompts_dispatches_to_correct_loader():
    """load_prompts dispatches by id; missing ids raise."""
    provider = _fixture_provider({("any", "test"): [{"question": "q", "answer": "a"}]})
    prompts = list(load_prompts("gsm8k", examples_provider=provider))
    assert prompts == ["q"]


def test_load_prompts_rejects_unknown_benchmark():
    with pytest.raises(ValueError, match="unknown benchmark id"):
        list(load_prompts("not-a-real-benchmark"))


# --------------------------------------------------------------------------- #
# Integration: prompts produced by the loaders flow into the index correctly.
# Without this test, we could ship a loader that yields strings the index
# normalizer chokes on (e.g. all-punctuation prompts).
# --------------------------------------------------------------------------- #
def test_loader_output_indexable_by_decontamination():
    from myllm.data.decontamination import DecontaminationConfig, DecontaminationIndex

    provider = _fixture_provider(
        {
            ("any", "test"): [
                {"problem": "Compute the integral of x squared from zero to one"},
            ]
        }
    )
    prompts = list(iter_math(examples_provider=provider))
    idx = DecontaminationIndex(DecontaminationConfig(ngram_size=5))
    idx.add_benchmark("math", prompts)
    matches = idx.scan_document(
        "Today's homework: compute the integral of x squared from zero to one"
    )
    assert "math" in matches
