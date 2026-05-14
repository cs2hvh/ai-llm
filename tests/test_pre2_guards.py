from __future__ import annotations

import pytest

from myllm_pre2.guards import reject_topk_kd_inputs


def test_pre2_guard_allows_ce_only_batches():
    reject_topk_kd_inputs()


def test_pre2_guard_rejects_heterogeneous_topk_kd_payload():
    with pytest.raises(ValueError, match="heterogeneous-tokenizer top-K"):
        reject_topk_kd_inputs(teacher_topk_logits=object(), teacher_topk_indices=object())


def test_pre2_guard_rejects_same_tokenizer_kd_until_implemented():
    with pytest.raises(ValueError, match="same-tokenizer top-K KD is not implemented"):
        reject_topk_kd_inputs(teacher_tokenizer_hash="abc", student_tokenizer_hash="abc")
