"""MILU adapter — Multitask Indian Language Understanding benchmark.

Reference: Verma et al. (AI4Bharat) 2024. The Indic equivalent of MMLU,
covering 11 Indian languages across academic, cultural and reasoning
subjects. Especially relevant for the MyLLM "sovereign hedge" (Path B)
strategy where Hindi is a primary secondary language sourced from
AI4Bharat/Sangraha.

HF dataset:  ``ai4bharat/MILU`` (one config per language code like
``hindi``, ``bengali``, ``tamil``, etc.).

Format:
    Each example has:
        question:     the question text
        option1..4:   four candidate answers
        answer:       the correct option as the *string* of one of the four
                      candidates (NOT a letter like A-D)
        domain:       subject area (e.g. "history", "physics")
        language:     language code

Prompt template (4-shot in-context):
    Q: {question_1}
    A. {option1_1}
    B. {option1_2}
    C. {option1_3}
    D. {option1_4}
    Answer: {gold_letter_1}

    Q: {question_2}
    ...

    Q: {target_question}
    A. ...
    Answer:

Scoring quirk vs MMLU-ProX:
    MILU stores the *content string* of the correct option, not the letter.
    Our adapter maps that string back to a letter at example-load time
    (taking the first option that matches verbatim). If no option matches
    we skip the row with a warning.
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

_LETTERS = string.ascii_uppercase[:4]  # A B C D
_VALID_ANSWERS = set(_LETTERS)

_PREFIX_RE = re.compile(
    r"^\s*(?:THE\s+)?(?:CORRECT\s+|FINAL\s+|RIGHT\s+)?"
    r"(?:ANSWER\s*(?:IS\s*)?|CHOICE\s+(?:IS\s+)?)?[:.()\[\]\s]*",
    re.IGNORECASE,
)

# Default = Hindi only; the most relevant subgroup for our project per
# `docs/playbook_alignment.md` S1 (Hindi from Sangraha at 4% of pretrain).
# Pass languages=("hindi", "bengali", ...) for broader Indic eval.
DEFAULT_LANGUAGES = ("hindi",)


class MILUBenchmark:
    """11-Indian-language multitask understanding benchmark.

    Test injection: pass ``examples_provider`` to bypass HF.
    """

    name = "milu"

    def __init__(
        self,
        *,
        languages: tuple[str, ...] = DEFAULT_LANGUAGES,
        n_shots: int = 4,
        hf_dataset: str = "ai4bharat/MILU",
        examples_provider: Any | None = None,
    ):
        self.languages = languages
        self.n_shots = n_shots
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
                "datasets library required for MILU; "
                "install with `pip install datasets`"
            ) from e
        ds = load_dataset(self.hf_dataset, name=language, split=split, streaming=False)
        for row in ds:
            yield row

    @staticmethod
    def _row_to_letter(row: dict) -> str | None:
        """Map MILU's content-string answer to a letter A-D.

        Returns None if no option matches the answer string (we skip these
        rather than guess).
        """
        options = [row["option1"], row["option2"], row["option3"], row["option4"]]
        gold = row["answer"]
        for i, opt in enumerate(options):
            if opt == gold:
                return _LETTERS[i]
        # Try a normalized match (some datasets have whitespace differences).
        gold_norm = (gold or "").strip()
        for i, opt in enumerate(options):
            if (opt or "").strip() == gold_norm:
                return _LETTERS[i]
        return None

    @staticmethod
    def _format_one_question(row: dict, include_answer: bool, gold_letter: str | None = None) -> str:
        question = row["question"]
        options = [row["option1"], row["option2"], row["option3"], row["option4"]]
        lines = [f"Q: {question}"]
        for i, opt in enumerate(options):
            lines.append(f"{_LETTERS[i]}. {opt}")
        if include_answer:
            assert gold_letter is not None
            lines.append(f"Answer: {gold_letter}")
        else:
            lines.append("Answer:")
        return "\n".join(lines)

    def _build_prompt(self, target_row: dict, shot_pairs: list[tuple[dict, str]]) -> str:
        blocks = [
            self._format_one_question(row, include_answer=True, gold_letter=letter)
            for row, letter in shot_pairs
        ]
        blocks.append(self._format_one_question(target_row, include_answer=False))
        return "\n\n".join(blocks)

    def load_examples(
        self,
        split: str = "test",
        sample_size: int | None = None,
        seed: int = 0,
    ) -> Iterator[EvalExample]:
        for lang in self.languages:
            rng = random.Random(seed + hash(lang))
            raw = list(self._iter_raw_examples(lang, split))
            # Resolve each row's gold letter; drop rows we can't match.
            resolved: list[tuple[dict, str]] = []
            n_skipped = 0
            for row in raw:
                letter = self._row_to_letter(row)
                if letter is None:
                    n_skipped += 1
                    continue
                resolved.append((row, letter))
            if n_skipped:
                log.warning("milu_rows_skipped_unmatched_answer", language=lang, n_skipped=n_skipped)
            if len(resolved) < self.n_shots + 1:
                log.warning(
                    "milu_language_too_small",
                    language=lang,
                    n_resolved=len(resolved),
                    needed=self.n_shots + 1,
                )
                continue
            rng.shuffle(resolved)
            shots = resolved[: self.n_shots]
            test_pairs = resolved[self.n_shots:]
            if sample_size is not None:
                test_pairs = test_pairs[:sample_size]
            for row, letter in test_pairs:
                prompt = self._build_prompt(row, shots)
                yield EvalExample(
                    prompt=prompt,
                    target_answer=letter,
                    metadata={
                        "language": lang,
                        "domain": row.get("domain", "n/a"),
                    },
                )

    # ------------------------------------------------------------------- #
    # Scoring — same heuristic as MMLU-ProX / Belebele
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
