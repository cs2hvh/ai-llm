"""MMLU-ProX adapter — multilingual MMLU-Pro across 29 languages.

Reference: Wang et al., "MMLU-ProX: A Multilingual Benchmark for Advanced
Large Language Model Evaluation", arXiv:2503.10497, 2025.

HF dataset:  ``MMLU-ProX/mmlu-prox`` (one config per language code,
e.g. ``en``, ``hi``, ``zh``, ``ar``, ``fr``, ``de``, ``es``).

Format:
    Each example is a multiple-choice question with 4-10 options
    (A, B, C, D, ... up to J). Answer is a single letter.

Prompt template (4-shot in-context, locked):
    Q: {question_1}
    A. {choice_1a}
    B. {choice_1b}
    ...
    Answer: {answer_1}

    Q: {question_2}
    ...

    Q: {target_question}
    A. ...
    Answer:

The model is expected to output a single letter; we extract the first
alphabetic character of its response and compare to the gold letter.
"""
from __future__ import annotations

import random
import re
import string
from collections.abc import Iterator
from typing import Any

from myllm.eval.types import EvalExample
from myllm.utils import get_logger

log = get_logger(__name__)

_LETTERS = string.ascii_uppercase
_VALID_ANSWERS = set("ABCDEFGHIJ")

# Default language coverage for our gate eval. The user can override.
DEFAULT_LANGUAGES = ("en", "hi", "es", "fr", "de", "zh", "ar")

# Strip common prefixes the model may emit before the answer letter.
_PREFIX_RE = re.compile(
    r"^\s*(?:THE\s+)?(?:CORRECT\s+|FINAL\s+|RIGHT\s+)?"
    r"(?:ANSWER\s*(?:IS\s*)?|CHOICE\s+(?:IS\s+)?)?[:.()\[\]\s]*",
    re.IGNORECASE,
)


class MMLUProXBenchmark:
    """29-language MMLU-Pro benchmark.

    By default we run on our 7 product languages (English + 6 secondaries).
    Pass ``languages=...`` to override.

    Test injection: pass ``examples_provider`` to bypass HF for tests.
    """

    name = "mmlu-prox"

    def __init__(
        self,
        *,
        languages: tuple[str, ...] = DEFAULT_LANGUAGES,
        n_shots: int = 4,
        hf_dataset: str = "li-lab/MMLU-ProX",
        examples_provider: Any | None = None,
    ):
        self.languages = languages
        self.n_shots = n_shots
        self.hf_dataset = hf_dataset
        # When examples_provider is set (in tests), we skip HF entirely.
        # Signature: provider(language, split) -> Iterable[dict-like-row].
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
                "datasets library required for MMLU-ProX; "
                "install with `pip install datasets`"
            ) from e
        ds = load_dataset(self.hf_dataset, name=language, split=split, streaming=False)
        for row in ds:
            yield row

    # ------------------------------------------------------------------- #
    # Prompt formatting
    # ------------------------------------------------------------------- #
    @staticmethod
    def _format_one_question(row: dict, include_answer: bool) -> str:
        """Render a question + choices block. Optionally append the answer."""
        question = row["question"]
        choices = row["options"]  # list[str], variable length 4-10
        lines = [f"Q: {question}"]
        for i, choice in enumerate(choices):
            lines.append(f"{_LETTERS[i]}. {choice}")
        if include_answer:
            answer_letter = _LETTERS[int(row["answer_index"])]
            lines.append(f"Answer: {answer_letter}")
        else:
            lines.append("Answer:")
        return "\n".join(lines)

    def _build_prompt(self, target_row: dict, shot_rows: list[dict]) -> str:
        blocks = [self._format_one_question(r, include_answer=True) for r in shot_rows]
        blocks.append(self._format_one_question(target_row, include_answer=False))
        return "\n\n".join(blocks)

    def load_examples(
        self,
        split: str = "test",
        sample_size: int | None = None,
        seed: int = 0,
    ) -> Iterator[EvalExample]:
        """Stream prompt/target pairs for every configured language.

        Per language:
          1. Pull (n_shots + sample_size or all) rows.
          2. Use first n_shots as fixed in-context exemplars.
          3. For each remaining row, build a 4-shot prompt + emit one
             ``EvalExample`` with metadata={"language": lang}.
        """
        for lang in self.languages:
            rng = random.Random(seed + hash(lang))
            rows = list(self._iter_raw_examples(lang, split))
            if len(rows) < self.n_shots + 1:
                log.warning(
                    "mmlu_prox_language_too_small",
                    language=lang,
                    n_rows=len(rows),
                    needed=self.n_shots + 1,
                )
                continue
            rng.shuffle(rows)
            shots = rows[: self.n_shots]
            test_rows = rows[self.n_shots:]
            if sample_size is not None:
                test_rows = test_rows[:sample_size]
            for r in test_rows:
                prompt = self._build_prompt(r, shots)
                target = _LETTERS[int(r["answer_index"])]
                yield EvalExample(
                    prompt=prompt,
                    target_answer=target,
                    metadata={"language": lang, "subject": r.get("subject", "n/a")},
                )

    # ------------------------------------------------------------------- #
    # Scoring
    # ------------------------------------------------------------------- #
    def score(self, prediction: str, example: EvalExample) -> bool:
        """Extract the answer letter from the prediction.

        Strategy (lm-eval-harness convention):
          1. Uppercase and strip common prefixes ("The answer is", "Answer:",
             whitespace, parens, etc.).
          2. If the first remaining character is a valid answer letter A-J,
             that's the prediction.
          3. Otherwise, return False (model didn't follow the format).

        This intentionally does NOT do "last letter A-J in the string" —
        that heuristic over-credits CoT responses that mention letters in
        passing ("B is wrong because A says X..."). For CoT-style models
        we'll add a separate extractor in a follow-up.
        """
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
