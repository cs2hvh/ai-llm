"""MMLU-Pro adapter — harder, English-only successor to MMLU.

Reference: Wang et al., "MMLU-Pro: A More Robust and Challenging Multi-Task
Language Understanding Benchmark", arXiv:2406.01574, 2024.

HF dataset: ``TIGER-Lab/MMLU-Pro``.

Format:
    Each example is a multiple-choice question with 4-10 options (A-J).
    Answer is a single letter. 14 subject categories ("biology", "law",
    "math", "psychology", etc.).

Prompt template (5-shot in-context, locked):
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

Scoring (lm-eval-harness convention): uppercase + strip prefixes;
first remaining character is the prediction. We do NOT scan for any
A-J letter in the string — that over-credits CoT chains that mention
letters in passing.

Round D6 / Layer 2 (2026-05-16): added so the release scorecard
benchmark name "mmlu-pro" no longer falls through to the placeholder
"non-empty output = success" scorer.
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


def _extract_choices(row: dict) -> list[str]:
    """Return the row's choices.

    HF schema is ``options: list[str]`` (4-10 items). We also accept
    the column-wise ``option_0..option_9`` shape that some MMLU-Pro
    mirrors use, since the test fixtures shouldn't have to know which
    storage style the dataset version uses.
    """
    if "options" in row and row["options"] is not None:
        return [c for c in row["options"] if c not in (None, "", "N/A")]
    out: list[str] = []
    for j in range(10):
        v = row.get(f"option_{j}")
        if v in (None, "", "N/A"):
            continue
        out.append(v)
    return out


def _answer_letter_from_row(row: dict) -> str:
    """Return the target answer letter (A-J)."""
    if "answer" in row and isinstance(row["answer"], str) and row["answer"]:
        a = row["answer"].strip().upper()
        if a in _VALID_ANSWERS:
            return a
    if "answer_index" in row and row["answer_index"] is not None:
        return _LETTERS[int(row["answer_index"])]
    raise ValueError(
        f"MMLU-Pro row missing both 'answer' (letter) and 'answer_index' (int): {row!r}"
    )


_PREFIX_RE = re.compile(
    r"^\s*(?:THE\s+)?(?:CORRECT\s+|FINAL\s+|RIGHT\s+)?"
    r"(?:ANSWER\s*(?:IS\s*)?|CHOICE\s+(?:IS\s+)?)?[:.()\[\]\s]*",
    re.IGNORECASE,
)


class MMLUProBenchmark:
    """English-only MMLU-Pro benchmark with proper letter-based scoring."""

    def __init__(
        self,
        hf_dataset: str = "TIGER-Lab/MMLU-Pro",
        split: str = "test",
        n_shots: int = 5,
        examples_provider: Any = None,
    ) -> None:
        self.name = "mmlu-pro"
        self.hf_dataset = hf_dataset
        self.split = split
        self.n_shots = int(n_shots)
        # Hook for tests — lets us inject a list of rows without touching HF.
        # Signature: provider(split) -> Iterable[dict-like-row].
        self._examples_provider = examples_provider

    def _iter_raw_examples(self, split: str) -> Iterator[dict]:
        if self._examples_provider is not None:
            yield from self._examples_provider(split)
            return
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError(
                "datasets library required for MMLU-Pro; "
                "install with `pip install datasets`"
            ) from e
        ds = load_dataset(self.hf_dataset, split=split, streaming=False)
        for row in ds:
            yield row

    @staticmethod
    def _format_one_question(row: dict, include_answer: bool) -> str:
        question = row["question"]
        choices = _extract_choices(row)
        lines = [f"Q: {question}"]
        for i, choice in enumerate(choices):
            lines.append(f"{_LETTERS[i]}. {choice}")
        if include_answer:
            lines.append(f"Answer: {_answer_letter_from_row(row)}")
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
        """Stream prompt/target pairs.

        1. Pull all rows.
        2. Use the first n_shots (after shuffle) as fixed in-context exemplars.
        3. For each remaining row, build an n_shot prompt + emit one
           ``EvalExample`` with metadata={"subject": ...}.
        """
        rng = random.Random(seed)
        rows = list(self._iter_raw_examples(split))
        if len(rows) < self.n_shots + 1:
            log.warning(
                "mmlu_pro_dataset_too_small",
                n_rows=len(rows), needed=self.n_shots + 1,
            )
            return
        rng.shuffle(rows)
        shots = rows[: self.n_shots]
        test_rows = rows[self.n_shots:]
        if sample_size is not None:
            test_rows = test_rows[:sample_size]
        for r in test_rows:
            target = _answer_letter_from_row(r)
            yield EvalExample(
                prompt=self._build_prompt(r, shots),
                target_answer=target,
                metadata={
                    "subject": r.get("category") or r.get("subject") or "n/a",
                },
            )

    def score(self, prediction: str, example: EvalExample) -> bool:
        """First A-J after prefix-strip == target letter."""
        pred = prediction.upper()
        stripped = _PREFIX_RE.sub("", pred, count=1).lstrip()
        if not stripped:
            return False
        first = stripped[0]
        if first not in _VALID_ANSWERS:
            return False
        return first == example.target_answer

    def subgroup_key(self, example: EvalExample) -> str:
        return example.metadata.get("subject", "n/a")
