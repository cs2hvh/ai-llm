"""Special-token constants and tokenizer-verification utility.

Single source of truth for the literal strings used as special tokens. Any
script that looks up a token id must reference these constants, never the
literal string. This eliminates a class of typo bugs (e.g.,
``"<|eos|>"`` vs ``"<|EOS|>"``) that would otherwise produce silent failures
months later when the tokenizer expects one and a script types the other.

Update flow when adding a token:
    1. Add the constant here (in :class:`SpecialTokens`).
    2. Add it to :data:`REQUIRED` if every tokenizer must include it,
       or :data:`OPTIONAL` otherwise.
    3. Add it to ``configs/tokenizer.yaml`` ``special_tokens`` list.
    4. Re-train the tokenizer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SpecialTokens:
    """Canonical string forms of every special token MyLLM uses."""

    BOS: str = "<|bos|>"
    EOS: str = "<|eos|>"
    PAD: str = "<|pad|>"
    UNK: str = "<|unk|>"  # required by Unigram tokenizer model
    IM_START: str = "<|im_start|>"
    IM_END: str = "<|im_end|>"
    TOOL_CALL: str = "<|tool_call|>"
    TOOL_RESULT: str = "<|tool_result|>"


# Tokens every tokenizer MUST contain. Verified at training-script startup.
REQUIRED: tuple[str, ...] = (
    SpecialTokens.BOS,
    SpecialTokens.EOS,
    SpecialTokens.PAD,
    SpecialTokens.UNK,
    SpecialTokens.IM_START,
    SpecialTokens.IM_END,
)

# Tokens that may be present (used by post-training tool-use tuning).
OPTIONAL: tuple[str, ...] = (
    SpecialTokens.TOOL_CALL,
    SpecialTokens.TOOL_RESULT,
)


def all_special_token_strings(reserved_slots: int = 0) -> list[str]:
    """Return REQUIRED + OPTIONAL + reserved ``<|extra_N|>`` slots.

    Used by the tokenizer trainer to seed ``BpeTrainer`` with the full
    special-token set in a deterministic order.
    """
    base = list(REQUIRED) + list(OPTIONAL)
    base += [f"<|extra_{i}|>" for i in range(reserved_slots)]
    return base


def verify_tokenizer_has_required(tokenizer: Any) -> None:
    """Raise ``ValueError`` if any REQUIRED token is missing from the tokenizer.

    Call this once at the top of any script that looks up token ids — it
    fails loudly on a misconfigured tokenizer instead of letting the script
    silently produce ``None`` ids that crash downstream.
    """
    missing = [t for t in REQUIRED if tokenizer.token_to_id(t) is None]
    if missing:
        raise ValueError(
            f"tokenizer is missing required special tokens: {missing}. "
            f"Either re-train the tokenizer with these in special_tokens, "
            f"or update myllm.data.special_tokens.REQUIRED."
        )


def get_required_ids(tokenizer: Any) -> dict[str, int]:
    """Return ``{token_string: id}`` for every required token. Verifies first."""
    verify_tokenizer_has_required(tokenizer)
    return {t: tokenizer.token_to_id(t) for t in REQUIRED}
