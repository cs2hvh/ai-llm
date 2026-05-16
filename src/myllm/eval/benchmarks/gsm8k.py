"""GSM8K adapter — grade-school math word problems.

Reference: Cobbe et al., "Training Verifiers to Solve Math Word Problems",
arXiv:2110.14168, 2021.

HF dataset: ``openai/gsm8k`` (config ``main``).

Format:
    {question: str, answer: str}. The answer is a CoT-shaped string
    ending in ``#### N`` where N is the final integer answer.

    Example:
        question: "Janet has 3 apples. She buys 2 more. How many?"
        answer:   "She had 3 apples. She bought 2 more, so 3+2=5. #### 5"

Prompt template (8-shot CoT, locked):
    Q: {question_1}
    A: {answer_1_full_CoT}

    Q: {question_2}
    A: ...

    Q: {target_question}
    A:

Scoring: parse the LAST ``#### N`` (or final integer) from the
prediction and compare to the gold integer. This matches the
GSM8K-standard "exact match on the final number" convention used by
EleutherAI lm-eval-harness, OpenAI's original eval, and most cited
papers. We do NOT do partial credit / sub-string matching.

Round D6 / Layer 2 (2026-05-16): added so the release scorecard
benchmark name "gsm8k" no longer falls through to the placeholder
"non-empty output = success" scorer.
"""
from __future__ import annotations

import random
import re
from collections.abc import Iterator
from typing import Any

from myllm.eval.types import EvalExample
from myllm.utils import get_logger

log = get_logger(__name__)

# Matches `#### -1234` or `#### 1234.5` (trailing decimals tolerated).
# The grade-school dataset's gold answers are all integers; we still
# accept floats from the model for robustness.
_FINAL_ANSWER_RE = re.compile(r"####\s*(-?\d+(?:\.\d+)?)")
# Fallback: last signed integer anywhere in the text. Used only when
# the model didn't produce the canonical `#### N` marker.
_ANY_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _gold_answer_from_row(row: dict) -> str:
    """Extract the canonical integer answer from a GSM8K row.

    Row schema: ``answer: str`` ending in ``#### N``. The portion
    before the marker is the CoT chain (not used for scoring).
    """
    a = row.get("answer", "")
    m = _FINAL_ANSWER_RE.search(a)
    if not m:
        raise ValueError(
            f"GSM8K row answer missing '#### N' marker: {a[:200]!r}"
        )
    return m.group(1)


def _normalize_number(s: str) -> str:
    """Drop trailing .0 / .00 so '5.0' == '5' for exact-match scoring."""
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
        return s
    except ValueError:
        return s


def _extract_predicted_number(prediction: str) -> str | None:
    """Extract the model's final numeric answer.

    Priority:
      1. Last ``#### N`` marker (CoT convention).
      2. Last signed number anywhere in the response.
      3. None — model produced no number.
    """
    markers = _FINAL_ANSWER_RE.findall(prediction)
    if markers:
        return _normalize_number(markers[-1])
    nums = _ANY_NUMBER_RE.findall(prediction)
    if nums:
        return _normalize_number(nums[-1])
    return None


class GSM8KBenchmark:
    """GSM8K with 8-shot CoT prompting + final-number-match scoring."""

    def __init__(
        self,
        hf_dataset: str = "openai/gsm8k",
        config_name: str = "main",
        split: str = "test",
        n_shots: int = 8,
        examples_provider: Any = None,
    ) -> None:
        self.name = "gsm8k"
        self.hf_dataset = hf_dataset
        self.config_name = config_name
        self.split = split
        self.n_shots = int(n_shots)
        self._examples_provider = examples_provider

    def _iter_raw_examples(self, split: str) -> Iterator[dict]:
        if self._examples_provider is not None:
            yield from self._examples_provider(split)
            return
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError(
                "datasets library required for GSM8K; "
                "install with `pip install datasets`"
            ) from e
        ds = load_dataset(
            self.hf_dataset, name=self.config_name,
            split=split, streaming=False,
        )
        for row in ds:
            yield row

    @staticmethod
    def _format_one_question(row: dict, include_answer: bool) -> str:
        lines = [f"Q: {row['question']}"]
        if include_answer:
            lines.append(f"A: {row['answer']}")
        else:
            lines.append("A:")
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

        We use shots from the TRAIN split (canonical GSM8K eval recipe —
        prevents leakage between in-context exemplars and held-out
        items). Falls back to using the start of the requested split
        if the train split isn't available (e.g., test fixtures with
        only one split).
        """
        try:
            shot_rows = list(self._iter_raw_examples("train"))
        except Exception as e:  # noqa: BLE001
            log.warning(
                "gsm8k_train_split_unavailable",
                error=str(e),
                msg="falling back to using head of eval split for shots",
            )
            shot_rows = []

        rng = random.Random(seed)
        if shot_rows:
            rng.shuffle(shot_rows)
            shots = shot_rows[: self.n_shots]
            test_rows = list(self._iter_raw_examples(split))
        else:
            all_rows = list(self._iter_raw_examples(split))
            if len(all_rows) < self.n_shots + 1:
                log.warning(
                    "gsm8k_dataset_too_small",
                    n_rows=len(all_rows), needed=self.n_shots + 1,
                )
                return
            rng.shuffle(all_rows)
            shots = all_rows[: self.n_shots]
            test_rows = all_rows[self.n_shots:]

        if sample_size is not None:
            test_rows = test_rows[:sample_size]

        for r in test_rows:
            gold = _gold_answer_from_row(r)
            yield EvalExample(
                prompt=self._build_prompt(r, shots),
                target_answer=_normalize_number(gold),
                metadata={"task_id": r.get("task_id", "n/a")},
            )

    def score(self, prediction: str, example: EvalExample) -> bool:
        predicted = _extract_predicted_number(prediction)
        if predicted is None:
            return False
        return predicted == example.target_answer

    def subgroup_key(self, example: EvalExample) -> str:
        # GSM8K is single-task; everything is in one bucket.
        return "all"
