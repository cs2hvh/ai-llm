"""Prompt-text loaders for benchmark decontamination.

For decontamination purposes we don't need the full Benchmark adapter
machinery (prompt formatting templates, scoring logic, subgroup keys).
All that matters is: for each benchmark, yield the raw question/problem
text whose n-grams should be excluded from the pretrain corpus.

Each loader returns an iterator of strings. The strings are passed
through ``DecontaminationIndex.add_benchmark`` which case-folds and
strips punctuation before n-gram hashing — so we don't need to do
any prompt normalization here. Just get the raw text out of the HF
row and hand it over.

Benchmarks covered (matches the v1 eval gate list in model_card_v1_template.md
plus the ones called out by the second external reviewer 2026-05-12):
  - mmlu-pro       — TIGER-Lab/MMLU-Pro, MIT
  - humaneval-plus — evalplus/humanevalplus, MIT
  - mbpp-plus      — evalplus/mbppplus, MIT
  - gsm8k          — openai/gsm8k, MIT
  - math           — HuggingFaceH4/MATH-500, MIT (Hendrycks et al. competition_math)
  - mgsm           — juletxara/mgsm, MIT
  - bbh            — maveriq/bigbenchhard, Apache-2.0
  - ifeval         — google/IFEval, Apache-2.0

Excluded:
  - LiveCodeBench: release-versioned; we re-index per release in Phase C.
  - RULER: synthetic / generated at eval time; no static prompts to index.

The 3 existing Benchmark adapters (mmlu-prox, belebele, milu) live in
src/myllm/eval/benchmarks/ and continue to be used via
``extract_prompts_from_benchmark``. This module is for the rest.
"""
from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Callable

from myllm.utils import get_logger

log = get_logger(__name__)

# Type alias: an "examples_provider" is a function used in tests to bypass HF.
# It takes (dataset_id, split) and yields dict-like rows in the same shape
# the loader expects from datasets.load_dataset.
ExamplesProvider = Callable[[str, str], Iterator[dict]]


def _load_hf(
    dataset_id: str,
    split: str,
    *,
    name: str | None = None,
    streaming: bool = False,
    examples_provider: ExamplesProvider | None = None,
) -> Iterator[dict]:
    """Load rows from HF or from a test-injected provider.

    Centralising the import lets us defer the `datasets` import until
    actually needed (the lib is heavy + only required at index-build time,
    not at training-loop time).
    """
    if examples_provider is not None:
        yield from examples_provider(dataset_id, split)
        return
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "datasets library required for prompt loaders; "
            "install with `pip install datasets`"
        ) from e
    kwargs: dict[str, Any] = {"split": split, "streaming": streaming}
    if name is not None:
        kwargs["name"] = name
    ds = load_dataset(dataset_id, **kwargs)
    yield from ds


def _truncate_if_huge(text: str, *, max_chars: int = 8000) -> str:
    """Defensive cap on prompt size before n-gram hashing.

    Some benchmark prompts (BBH, RULER long-context) can be very long;
    we only need their distinctive n-grams, not the entire text. A
    multi-page prompt produces millions of n-grams, blowing up the index
    for no contamination-detection benefit (the first ~8000 chars
    already give plenty of distinguishing 13-grams).
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def iter_mmlu_pro(
    *,
    dataset_id: str = "TIGER-Lab/MMLU-Pro",
    split: str = "test",
    sample_size: int | None = None,
    examples_provider: ExamplesProvider | None = None,
) -> Iterator[str]:
    """MMLU-Pro: question + options string.

    Row schema: {question: str, options: list[str], answer: str, ...}.
    Yields the question plus the options so corpus copies of either
    the question OR the question+answer-choices are caught.
    """
    n = 0
    for row in _load_hf(dataset_id, split, examples_provider=examples_provider):
        question = (row.get("question") or "").strip()
        if not question:
            # Skip rows with an empty/whitespace question, regardless of
            # options — the question is what gets contaminated.
            continue
        options = row.get("options") or []
        text = question
        if options:
            text = text + " " + " ".join(options)
        yield _truncate_if_huge(text)
        n += 1
        if sample_size is not None and n >= sample_size:
            return


def iter_humaneval_plus(
    *,
    dataset_id: str = "evalplus/humanevalplus",
    split: str = "test",
    sample_size: int | None = None,
    examples_provider: ExamplesProvider | None = None,
) -> Iterator[str]:
    """HumanEval+: function signature + docstring.

    Row schema: {task_id: str, prompt: str, canonical_solution: str, ...}.
    The ``prompt`` field is the function signature + docstring — exactly
    what would appear verbatim in a contaminated corpus (Stack-Overflow
    answer, blog post, GitHub gist).
    """
    n = 0
    for row in _load_hf(dataset_id, split, examples_provider=examples_provider):
        text = row.get("prompt") or ""
        if text.strip():
            yield _truncate_if_huge(text)
            n += 1
            if sample_size is not None and n >= sample_size:
                return


def iter_mbpp_plus(
    *,
    dataset_id: str = "evalplus/mbppplus",
    split: str = "test",
    sample_size: int | None = None,
    examples_provider: ExamplesProvider | None = None,
) -> Iterator[str]:
    """MBPP+: natural-language problem statement.

    Row schema: {task_id, text/prompt: str, code: str, test_list: list, ...}.
    Some MBPP variants use ``text``, others ``prompt`` — try both.
    """
    n = 0
    for row in _load_hf(dataset_id, split, examples_provider=examples_provider):
        text = row.get("prompt") or row.get("text") or ""
        if text.strip():
            yield _truncate_if_huge(text)
            n += 1
            if sample_size is not None and n >= sample_size:
                return


def iter_gsm8k(
    *,
    dataset_id: str = "openai/gsm8k",
    split: str = "test",
    name: str = "main",
    sample_size: int | None = None,
    examples_provider: ExamplesProvider | None = None,
) -> Iterator[str]:
    """GSM8K: grade-school math word problems.

    Row schema: {question: str, answer: str}.
    Use the question only — the answer is what we're protecting.
    """
    n = 0
    for row in _load_hf(dataset_id, split, name=name, examples_provider=examples_provider):
        text = row.get("question") or ""
        if text.strip():
            yield _truncate_if_huge(text)
            n += 1
            if sample_size is not None and n >= sample_size:
                return


def iter_math(
    *,
    dataset_id: str = "HuggingFaceH4/MATH-500",
    split: str = "test",
    sample_size: int | None = None,
    examples_provider: ExamplesProvider | None = None,
) -> Iterator[str]:
    """MATH (Hendrycks et al. competition_math, 500-problem held-out).

    Row schema: {problem: str, solution: str, answer: str, level: str, type: str}.
    """
    n = 0
    for row in _load_hf(dataset_id, split, examples_provider=examples_provider):
        text = row.get("problem") or ""
        if text.strip():
            yield _truncate_if_huge(text)
            n += 1
            if sample_size is not None and n >= sample_size:
                return


def iter_mgsm(
    *,
    dataset_id: str = "juletxara/mgsm",
    split: str = "test",
    languages: tuple[str, ...] = ("en", "es", "fr", "de", "zh", "hi", "ar"),
    sample_size: int | None = None,
    examples_provider: ExamplesProvider | None = None,
) -> Iterator[str]:
    """MGSM: multilingual GSM8K (translated to 11 languages).

    Each language is a separate HF config. We default to our 7 product
    languages — same coverage as MMLU-ProX. Per language, sample_size is
    applied independently.

    Row schema (per config): {question: str, answer: str | None,
    answer_number: int, ...}.
    """
    for lang in languages:
        n = 0
        for row in _load_hf(
            dataset_id, split, name=lang, examples_provider=examples_provider
        ):
            text = row.get("question") or ""
            if text.strip():
                yield _truncate_if_huge(text)
                n += 1
                if sample_size is not None and n >= sample_size:
                    break


def iter_bbh(
    *,
    dataset_id: str = "maveriq/bigbenchhard",
    split: str = "train",
    subtasks: tuple[str, ...] | None = None,
    sample_size: int | None = None,
    examples_provider: ExamplesProvider | None = None,
) -> Iterator[str]:
    """BBH (Big-Bench Hard): 23 reasoning subtasks.

    Each subtask is its own config — when ``subtasks`` is None we pull
    every one. Row schema (per subtask): {input: str, target: str}.

    BBH ships with split="train" only (no test split); the convention is
    that the entire dataset is the held-out gate.
    """
    if subtasks is None:
        subtasks = _BBH_DEFAULT_SUBTASKS
    for sub in subtasks:
        n = 0
        for row in _load_hf(
            dataset_id, split, name=sub, examples_provider=examples_provider
        ):
            text = row.get("input") or ""
            if text.strip():
                yield _truncate_if_huge(text)
                n += 1
                if sample_size is not None and n >= sample_size:
                    break


_BBH_DEFAULT_SUBTASKS: tuple[str, ...] = (
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "disambiguation_qa",
    "dyck_languages",
    "formal_fallacies",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "logical_deduction_three_objects",
    "movie_recommendation",
    "multistep_arithmetic_two",
    "navigate",
    "object_counting",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "sports_understanding",
    "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
    "web_of_lies",
    "word_sorting",
)


def iter_ifeval(
    *,
    dataset_id: str = "google/IFEval",
    split: str = "train",
    sample_size: int | None = None,
    examples_provider: ExamplesProvider | None = None,
) -> Iterator[str]:
    """IFEval: instruction-following eval (Google).

    IFEval is "train" split only — the entire dataset is held-out gate
    material. Row schema: {key: int, prompt: str, instruction_id_list: list,
    kwargs: list}.
    """
    n = 0
    for row in _load_hf(dataset_id, split, examples_provider=examples_provider):
        text = row.get("prompt") or ""
        if text.strip():
            yield _truncate_if_huge(text)
            n += 1
            if sample_size is not None and n >= sample_size:
                return


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
# Keyed by the benchmark id used in build_decontamination_index.py. Each
# value is a zero-arg-with-kwargs loader; the build script passes
# (split, sample_size, examples_provider) through.
PROMPT_LOADERS: dict[str, Callable[..., Iterator[str]]] = {
    "mmlu-pro": iter_mmlu_pro,
    "humaneval-plus": iter_humaneval_plus,
    "mbpp-plus": iter_mbpp_plus,
    "gsm8k": iter_gsm8k,
    "math": iter_math,
    "mgsm": iter_mgsm,
    "bbh": iter_bbh,
    "ifeval": iter_ifeval,
}


def load_prompts(
    benchmark_id: str,
    *,
    split: str | None = None,
    sample_size: int | None = None,
    examples_provider: ExamplesProvider | None = None,
) -> Iterator[str]:
    """Dispatch helper: yield prompt strings for ``benchmark_id``.

    ``split=None`` (default) means: use the per-loader default split
    (e.g. ifeval defaults to "train" because it has no test split;
    gsm8k defaults to "test"). Pass an explicit ``split`` to override.

    2026-05-13 fix: previously this had ``split="test"`` as its OWN
    default which silently overrode each loader's per-benchmark
    default, breaking ifeval (train-only) and BBH (train-only).

    Raises ValueError for unknown ids so build_decontamination_index.py
    fails fast at sweep-launch instead of silently producing an
    incomplete index.
    """
    if benchmark_id not in PROMPT_LOADERS:
        raise ValueError(
            f"unknown benchmark id: {benchmark_id!r}. "
            f"Known: {sorted(PROMPT_LOADERS)}"
        )
    loader = PROMPT_LOADERS[benchmark_id]
    kwargs: dict[str, Any] = {
        "sample_size": sample_size,
        "examples_provider": examples_provider,
    }
    if split is not None:
        kwargs["split"] = split
    return loader(**kwargs)
