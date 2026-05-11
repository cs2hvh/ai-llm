"""Belebele adapter — multilingual reading comprehension across 122 languages.

Reference: Bandarkar et al., "The Belebele Benchmark: a Parallel Reading
Comprehension Dataset in 122 Language Variants", 2024.

HF dataset:  ``facebook/belebele``. One config per language variant
(e.g. ``eng_Latn``, ``hin_Deva``, ``arb_Arab``).

Format:
    Each example is a FLORES passage + a multiple-choice question with
    exactly 4 candidate answers (``mc_answer1`` … ``mc_answer4``) and a
    ``correct_answer_num`` (1-indexed string).

Prompt template (0-shot by default — Belebele is hard and few-shot doesn't
help below frontier scale; we'll add few-shot if pilot motivates it):

    Passage: {flores_passage}
    Question: {question}
    A. {mc_answer1}
    B. {mc_answer2}
    C. {mc_answer3}
    D. {mc_answer4}
    Answer:

Same letter-extraction heuristic as MMLU-ProX.
"""
from __future__ import annotations

import re
import string
from collections.abc import Iterator
from typing import Any

from myllm.eval.types import EvalExample
from myllm.utils import get_logger

log = get_logger(__name__)

_LETTERS = string.ascii_uppercase[:4]  # only A B C D for Belebele
_VALID_ANSWERS = set(_LETTERS)

# Same prefix-stripping heuristic as MMLU-ProX; see mmlu_prox.py docstring.
_PREFIX_RE = re.compile(
    r"^\s*(?:THE\s+)?(?:CORRECT\s+|FINAL\s+|RIGHT\s+)?"
    r"(?:ANSWER\s*(?:IS\s*)?|CHOICE\s+(?:IS\s+)?)?[:.()\[\]\s]*",
    re.IGNORECASE,
)

# Default language coverage = our 7 product languages mapped to Belebele
# config names (Belebele uses FLORES-200 BCP-47 codes; lookups below).
# We focus on the languages our model is trained on; the full 122-language
# eval is available by passing ``languages=...`` explicitly.
DEFAULT_LANGUAGES = (
    "eng_Latn",  # English
    "hin_Deva",  # Hindi (Devanagari)
    "spa_Latn",  # Spanish
    "fra_Latn",  # French
    "deu_Latn",  # German
    "zho_Hans",  # Chinese (Simplified)
    "arb_Arab",  # Modern Standard Arabic
)


class BelebeleBenchmark:
    """122-language reading-comprehension benchmark.

    Test injection: pass ``examples_provider`` to bypass HF for tests.
    """

    name = "belebele"

    def __init__(
        self,
        *,
        languages: tuple[str, ...] = DEFAULT_LANGUAGES,
        hf_dataset: str = "facebook/belebele",
        examples_provider: Any | None = None,
    ):
        self.languages = languages
        self.hf_dataset = hf_dataset
        self._examples_provider = examples_provider

    # ------------------------------------------------------------------- #
    # Loading
    # ------------------------------------------------------------------- #
    def _iter_raw_examples(self, language: str, split: str) -> Iterator[dict]:
        if self._examples_provider is not None:
            yield from self._examples_provider(language, split)
            return
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError(
                "datasets library required for Belebele; "
                "install with `pip install datasets`"
            ) from e
        ds = load_dataset(self.hf_dataset, name=language, split=split, streaming=False)
        for row in ds:
            yield row

    @staticmethod
    def _format_prompt(row: dict) -> tuple[str, str]:
        """Render the prompt + return the gold answer letter.

        ``row`` is expected to have:
            flores_passage, question, mc_answer1..4, correct_answer_num
        """
        passage = row["flores_passage"]
        question = row["question"]
        choices = [row["mc_answer1"], row["mc_answer2"], row["mc_answer3"], row["mc_answer4"]]
        # correct_answer_num is a 1-indexed string; convert to 0-indexed letter.
        gold_idx = int(row["correct_answer_num"]) - 1
        prompt = (
            f"Passage: {passage}\n"
            f"Question: {question}\n"
            + "\n".join(f"{_LETTERS[i]}. {c}" for i, c in enumerate(choices))
            + "\nAnswer:"
        )
        return prompt, _LETTERS[gold_idx]

    def load_examples(
        self,
        split: str = "test",
        sample_size: int | None = None,
        seed: int = 0,
    ) -> Iterator[EvalExample]:
        for lang in self.languages:
            rows = list(self._iter_raw_examples(lang, split))
            if not rows:
                log.warning("belebele_empty_language", language=lang)
                continue
            if sample_size is not None:
                rows = rows[:sample_size]
            for r in rows:
                try:
                    prompt, gold = self._format_prompt(r)
                except (KeyError, ValueError) as e:
                    log.warning("belebele_skipped_malformed_row", language=lang, error=str(e))
                    continue
                yield EvalExample(
                    prompt=prompt,
                    target_answer=gold,
                    metadata={"language": lang},
                )

    # ------------------------------------------------------------------- #
    # Scoring — shared heuristic with MMLU-ProX
    # ------------------------------------------------------------------- #
    def score(self, prediction: str, example: EvalExample) -> bool:
        pred = prediction.upper()
        stripped = _PREFIX_RE.sub("", pred, count=1).lstrip()
        if not stripped:
            return False
        first = stripped[0]
        if first not in _VALID_ANSWERS:
            return False
        return first == example.target_answer

    def subgroup_key(self, example: EvalExample) -> str:
        return example.metadata.get("language", "unknown")
