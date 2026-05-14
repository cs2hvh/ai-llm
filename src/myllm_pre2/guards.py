"""Fail-closed guards for pre-2 training.

Pre-2 intentionally starts from a plain next-token CE baseline. Anything that
would quietly reintroduce the old heterogeneous-tokenizer top-K KD path should
stop at configuration or batch-validation time.
"""
from __future__ import annotations

from typing import Any


HETEROGENEOUS_TOPK_KD = "heterogeneous_tokenizer_topk_logit_kd"


def reject_topk_kd_inputs(
    *,
    teacher_topk_logits: Any | None = None,
    teacher_topk_indices: Any | None = None,
    teacher_tokenizer_hash: str | None = None,
    student_tokenizer_hash: str | None = None,
) -> None:
    """Reject top-K distillation inputs unless a future safe path is explicit.

    Same-tokenizer KD is allowed by the architecture decision as a future
    method, but there is no pre-2 implementation yet that proves identical
    tokenizers, loss semantics, and eval benefit. The baseline therefore
    rejects all top-K KD tensors.
    """
    has_teacher_payload = teacher_topk_logits is not None or teacher_topk_indices is not None
    has_tokenizer_claim = teacher_tokenizer_hash is not None or student_tokenizer_hash is not None
    if not has_teacher_payload and not has_tokenizer_claim:
        return

    if teacher_tokenizer_hash and student_tokenizer_hash and teacher_tokenizer_hash == student_tokenizer_hash:
        raise ValueError(
            "Pre-2 same-tokenizer top-K KD is not implemented. Use CE-only "
            "training or add an explicit same-tokenizer KD path with tests."
        )

    raise ValueError(
        "Pre-2 rejects heterogeneous-tokenizer top-K logit KD. Teacher token "
        "IDs are not student token IDs; use teacher-generated text or an "
        "explicit same-tokenizer KD implementation instead."
    )
