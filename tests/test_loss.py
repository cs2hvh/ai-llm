"""Loss-function correctness tests.

The CE path was refactored on 2026-05-12 to gather log-probs at the label
positions instead of one-hot-ing and summing. These tests lock in:

  1. The gather CE matches the textbook formula ``-log p(label)``.
  2. It matches the *legacy* one-hot + sum reference within numerical noise.
  3. ``ignore_index`` and ``loss_mask`` zero out the right positions.
  4. ``z_loss`` is correctly the mean of ``log_z**2``.
  5. ``kl_div_topk_loss`` and ``multi_teacher_kl_loss`` agree with manual KL.

The reference one-hot path is recomputed inside the tests (rather than
imported from a legacy module) so we don't have to keep the old code
around just to test against.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

# Force CPU + JAX backend so the tests run anywhere without a GPU/TF dep.
os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from myllm.training.loss import (  # noqa: E402
    cross_entropy_with_z_loss,
    distillation_mixed_loss,
    kl_div_topk_loss,
    multi_teacher_kl_loss,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _legacy_one_hot_ce_reference(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Reference CE via one-hot + sum (the pre-refactor formulation).

    Returns per-token NLL (no z-loss, no mask) — for direct equivalence
    against the gather path.
    """
    log_z = np.log(np.sum(np.exp(logits - logits.max(axis=-1, keepdims=True)), axis=-1)) \
        + logits.max(axis=-1)
    log_softmax = logits - log_z[..., None]
    one_hot = np.eye(logits.shape[-1])[labels]
    return -np.sum(one_hot * log_softmax, axis=-1)


def _textbook_ce_reference(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Reference CE via direct ``-log p(label)``. The truth."""
    log_z = np.log(np.sum(np.exp(logits - logits.max(axis=-1, keepdims=True)), axis=-1)) \
        + logits.max(axis=-1)
    log_softmax = logits - log_z[..., None]
    # Gather along last axis: for each (b, s), pick log_softmax[b, s, labels[b, s]]
    b, s, v = logits.shape
    bs_idx, s_idx = np.meshgrid(np.arange(b), np.arange(s), indexing="ij")
    return -log_softmax[bs_idx, s_idx, labels]


# --------------------------------------------------------------------------- #
# Gather-CE equivalence
# --------------------------------------------------------------------------- #
class TestCrossEntropyEquivalence:
    def test_gather_ce_matches_textbook(self):
        rng = np.random.default_rng(0)
        logits = rng.standard_normal((2, 5, 17)).astype(np.float32)
        labels = rng.integers(0, 17, size=(2, 5)).astype(np.int32)

        loss, metrics = cross_entropy_with_z_loss(logits, labels, z_loss_coef=0.0)
        ce_ours = float(metrics["ce"])
        ce_truth = float(_textbook_ce_reference(logits, labels).mean())
        assert np.isclose(ce_ours, ce_truth, atol=1e-5), (ce_ours, ce_truth)

    def test_gather_ce_matches_legacy_one_hot(self):
        # Per-token NLL identical to the old one-hot * log_softmax reduction.
        rng = np.random.default_rng(1)
        logits = rng.standard_normal((3, 4, 19)).astype(np.float32)
        labels = rng.integers(0, 19, size=(3, 4)).astype(np.int32)

        _, metrics = cross_entropy_with_z_loss(logits, labels, z_loss_coef=0.0)
        ce_ours = float(metrics["ce"])
        ce_legacy = float(_legacy_one_hot_ce_reference(logits, labels).mean())
        assert np.isclose(ce_ours, ce_legacy, atol=1e-5), (ce_ours, ce_legacy)

    def test_z_loss_is_logsumexp_squared_mean(self):
        rng = np.random.default_rng(2)
        logits = rng.standard_normal((1, 3, 11)).astype(np.float32)
        labels = rng.integers(0, 11, size=(1, 3)).astype(np.int32)

        _, metrics = cross_entropy_with_z_loss(logits, labels, z_loss_coef=1.0)
        log_z = np.log(
            np.sum(np.exp(logits - logits.max(axis=-1, keepdims=True)), axis=-1)
        ) + logits.max(axis=-1)
        expected_zloss = float(np.mean(log_z * log_z))
        assert np.isclose(float(metrics["z_loss"]), expected_zloss, atol=1e-5)

    def test_total_loss_combines_ce_plus_zloss(self):
        rng = np.random.default_rng(3)
        logits = rng.standard_normal((1, 4, 8)).astype(np.float32)
        labels = rng.integers(0, 8, size=(1, 4)).astype(np.int32)

        z_coef = 0.05
        total, metrics = cross_entropy_with_z_loss(
            logits, labels, z_loss_coef=z_coef
        )
        expected = float(metrics["ce"]) + z_coef * float(metrics["z_loss"])
        assert np.isclose(float(total), expected, atol=1e-5)


# --------------------------------------------------------------------------- #
# Ignore index + loss mask
# --------------------------------------------------------------------------- #
class TestIgnoreIndexAndMask:
    def test_ignore_index_zeros_those_positions(self):
        logits = np.zeros((1, 4, 5), dtype=np.float32)
        # All positions equal logits -> uniform -log(1/5) per token.
        labels = np.array([[0, 1, 2, 3]], dtype=np.int32)
        # No ignore: average over 4 positions.
        _, m_full = cross_entropy_with_z_loss(logits, labels, z_loss_coef=0.0)
        # Ignore index 2 — drop the 3rd position; average over 3 tokens.
        _, m_ig = cross_entropy_with_z_loss(
            logits, labels, ignore_index=2, z_loss_coef=0.0
        )
        # Both should produce the same per-token CE (uniform logits),
        # but with different denominators. CE-per-kept-token is the same.
        assert np.isclose(float(m_full["ce"]), float(m_ig["ce"]), atol=1e-6)

    def test_all_ignored_safe_against_div_zero(self):
        # All labels equal ignore_index — denom would be 0; the code clamps
        # to 1.0 so loss is well-defined (returns sum/1.0 = 0 since weight=0).
        # Label values must be in [0, vocab) — the gather-CE path (post
        # 2026-05-12 refactor) is not OOB-safe by design, unlike the legacy
        # one_hot path which silently returned 0 for out-of-vocab labels.
        # In production all labels come from the tokenizer so they are
        # always in-range; an OOB label would have been a bug, not a safe
        # value to drop.
        logits = np.zeros((1, 3, 4), dtype=np.float32)
        labels = np.array([[3, 3, 3]], dtype=np.int32)
        total, m = cross_entropy_with_z_loss(
            logits, labels, ignore_index=3, z_loss_coef=0.0
        )
        assert float(m["ce"]) == 0.0
        assert np.isfinite(float(total))

    def test_loss_mask_combines_with_ignore_index(self):
        # Two positions: one masked by ignore_index, one by loss_mask.
        rng = np.random.default_rng(4)
        logits = rng.standard_normal((1, 2, 6)).astype(np.float32)
        labels = np.array([[3, 5]], dtype=np.int32)
        # Mask out position 1; ignore_index unrelated to current labels.
        loss_mask = np.array([[1, 0]], dtype=np.int32)
        _, m = cross_entropy_with_z_loss(
            logits, labels, ignore_index=999, loss_mask=loss_mask, z_loss_coef=0.0
        )
        # Only position 0 contributes.
        expected_nll = float(_textbook_ce_reference(logits, labels)[0, 0])
        assert np.isclose(float(m["ce"]), expected_nll, atol=1e-5)


# --------------------------------------------------------------------------- #
# Top-K KL distillation
# --------------------------------------------------------------------------- #
class TestTopKKL:
    def test_topk_kl_zero_when_distributions_identical(self):
        # If we set teacher_topk_logits = student_at_topk and indices align,
        # then teacher and student distributions over top-K match → KL = 0.
        rng = np.random.default_rng(5)
        student_logits = rng.standard_normal((1, 2, 10)).astype(np.float32)
        # Use the top-K argmax indices of the student so distributions align.
        k = 4
        topk_idx = np.argpartition(-student_logits, k, axis=-1)[..., :k].astype(np.int32)
        topk_logits = np.take_along_axis(student_logits, topk_idx, axis=-1)

        kl = float(kl_div_topk_loss(student_logits, topk_logits, topk_idx))
        assert np.isclose(kl, 0.0, atol=1e-5), kl

    def test_topk_kl_nonnegative(self):
        # KL is always >= 0.
        rng = np.random.default_rng(6)
        student_logits = rng.standard_normal((2, 3, 12)).astype(np.float32)
        k = 5
        # Random (not student-aligned) top-K indices.
        topk_idx = rng.integers(0, 12, size=(2, 3, k)).astype(np.int32)
        topk_logits = rng.standard_normal((2, 3, k)).astype(np.float32)

        kl = float(kl_div_topk_loss(student_logits, topk_logits, topk_idx))
        assert kl >= -1e-6, kl  # tiny negative is numerical noise

    def test_multi_teacher_kl_averages_correctly(self):
        # Two teachers — same student. Multi-teacher average should equal
        # the unweighted mean of the two per-teacher KLs.
        rng = np.random.default_rng(7)
        student_logits = rng.standard_normal((1, 2, 8)).astype(np.float32)
        k = 3
        t1_idx = rng.integers(0, 8, size=(1, 2, k)).astype(np.int32)
        t1_log = rng.standard_normal((1, 2, k)).astype(np.float32)
        t2_idx = rng.integers(0, 8, size=(1, 2, k)).astype(np.int32)
        t2_log = rng.standard_normal((1, 2, k)).astype(np.float32)

        kl1 = float(kl_div_topk_loss(student_logits, t1_log, t1_idx))
        kl2 = float(kl_div_topk_loss(student_logits, t2_log, t2_idx))
        expected = 0.5 * (kl1 + kl2)

        stacked_log = np.stack([t1_log, t2_log], axis=0)
        stacked_idx = np.stack([t1_idx, t2_idx], axis=0)
        avg = float(
            multi_teacher_kl_loss(student_logits, stacked_log, stacked_idx)
        )
        assert np.isclose(avg, expected, atol=1e-5), (avg, expected)


# --------------------------------------------------------------------------- #
# Mixed CE + distillation
# --------------------------------------------------------------------------- #
class TestDistillationMixedLoss:
    def test_collapses_to_ce_when_no_teacher(self):
        rng = np.random.default_rng(8)
        logits = rng.standard_normal((1, 3, 7)).astype(np.float32)
        labels = rng.integers(0, 7, size=(1, 3)).astype(np.int32)

        total, metrics = distillation_mixed_loss(
            logits, labels,
            teacher_topk_logits_per_teacher=None,
            teacher_topk_indices_per_teacher=None,
            z_loss_coef=0.0,
        )
        _, ce_metrics = cross_entropy_with_z_loss(logits, labels, z_loss_coef=0.0)
        assert np.isclose(float(total), float(ce_metrics["ce"]), atol=1e-5)
        assert float(metrics["kl"]) == 0.0
        assert float(metrics["alpha"]) == 1.0

    def test_alpha_one_equals_pure_ce_even_with_teacher(self):
        rng = np.random.default_rng(9)
        logits = rng.standard_normal((1, 2, 6)).astype(np.float32)
        labels = rng.integers(0, 6, size=(1, 2)).astype(np.int32)
        t_idx = rng.integers(0, 6, size=(1, 1, 2, 3)).astype(np.int32)
        t_log = rng.standard_normal((1, 1, 2, 3)).astype(np.float32)

        total, metrics = distillation_mixed_loss(
            logits, labels, t_log, t_idx, alpha=1.0, z_loss_coef=0.0
        )
        _, ce_metrics = cross_entropy_with_z_loss(logits, labels, z_loss_coef=0.0)
        # CE term dominates when alpha=1.
        assert np.isclose(float(total), float(ce_metrics["ce"]), atol=1e-5)
