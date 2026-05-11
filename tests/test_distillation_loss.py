"""Regression tests for distillation losses — R0 from 2026-05-11 dossier.

Validates the mechanics of:
  - ``kl_div_topk_loss``           — single-teacher top-K KL
  - ``multi_teacher_kl_loss``      — averaged across teachers
  - ``distillation_mixed_loss``    — α·CE + (1-α)·KL, with graceful
                                     fallback to plain CE when no teacher

These are pure-math tests; no actual teacher model is loaded. We
hand-craft synthetic logits and verify the loss has the expected
properties (zero when distributions match, positive otherwise, etc.).

End-to-end testing with cached teacher logits is in a follow-up PR
once `scripts/cache_teacher_logits.py` lands.
"""
from __future__ import annotations

import math

import pytest

keras = pytest.importorskip("keras")
ops = keras.ops

from myllm.training.loss import (  # noqa: E402
    cross_entropy_with_z_loss,
    distillation_mixed_loss,
    kl_div_topk_loss,
    multi_teacher_kl_loss,
)

# Smaller-than-production for fast tests.
_V = 32   # vocab
_K = 4    # top-K
_B = 2    # batch
_S = 3    # seq
_T = 3    # teachers


# --------------------------------------------------------------------------- #
# kl_div_topk_loss
# --------------------------------------------------------------------------- #
class TestKlDivTopK:
    def _matched_pair(self):
        """Make student and teacher logits where the teacher's top-K is
        a subset of the student's logits at the same indices.
        """
        # Random student logits.
        student = keras.random.normal((_B, _S, _V), seed=0)
        # Teacher's top-K indices: pick any K positions.
        # Use the student's own top-K so student and teacher *can* agree.
        # ops.top_k returns (values, indices).
        teacher_indices = ops.cast(
            ops.argsort(-student, axis=-1)[..., :_K], "int32"
        )
        # Teacher's logits at those positions = student's values (so
        # distributions match → KL should be zero).
        teacher_logits = ops.take_along_axis(student, teacher_indices, axis=-1)
        return student, teacher_logits, teacher_indices

    def test_zero_when_distributions_match(self):
        student, teacher_logits, teacher_indices = self._matched_pair()
        kl = kl_div_topk_loss(student, teacher_logits, teacher_indices)
        # Student's distribution restricted to teacher's top-K positions
        # is exactly the teacher's distribution → KL = 0.
        assert float(kl) < 1.0e-5, f"expected zero KL, got {float(kl)}"

    def test_positive_when_distributions_differ(self):
        student, teacher_logits, teacher_indices = self._matched_pair()
        # Perturb teacher logits — distributions no longer match.
        teacher_logits_perturbed = teacher_logits + ops.cast(
            keras.random.normal(ops.shape(teacher_logits), seed=1), teacher_logits.dtype
        ) * 2.0
        kl = kl_div_topk_loss(student, teacher_logits_perturbed, teacher_indices)
        assert float(kl) > 1.0e-3, f"expected positive KL, got {float(kl)}"

    def test_temperature_softens_teacher(self):
        """Higher temperature → smoother teacher distribution → smaller KL
        when student is uniform-ish."""
        # Pick a student near uniform logits.
        student = ops.zeros((1, 1, _V))
        # Teacher: a peaked distribution.
        teacher_indices = ops.cast(ops.arange(_K)[None, None, :], "int32")
        teacher_logits = ops.array([[[10.0, 5.0, 1.0, 0.0]]], dtype="float32")
        kl_t1 = float(kl_div_topk_loss(student, teacher_logits, teacher_indices, temperature=1.0))
        kl_t4 = float(kl_div_topk_loss(student, teacher_logits, teacher_indices, temperature=4.0))
        # T=4 softens the teacher (more uniform) → matches the uniform-ish
        # student better → smaller KL.
        assert kl_t4 < kl_t1, f"expected T=4 KL < T=1 KL; got {kl_t4} vs {kl_t1}"

    def test_ignore_index_masks_padding(self):
        student, teacher_logits, teacher_indices = self._matched_pair()
        # Make labels for masking; mark some positions as padding.
        labels = ops.zeros((_B, _S), dtype="int32")
        # Without padding mask: KL ≈ 0 (matched).
        kl_no_mask = float(kl_div_topk_loss(student, teacher_logits, teacher_indices))
        assert kl_no_mask < 1.0e-5

        # With all positions ignored, the function should still return a
        # finite number (denom clamped to 1).
        kl_all_ignored = float(kl_div_topk_loss(
            student, teacher_logits, teacher_indices,
            ignore_index=0, labels=labels,
        ))
        # With all labels == ignore_index, the weight is all-zero, so the
        # numerator is zero too → result is zero.
        assert kl_all_ignored == pytest.approx(0.0, abs=1.0e-5)


# --------------------------------------------------------------------------- #
# multi_teacher_kl_loss
# --------------------------------------------------------------------------- #
class TestMultiTeacherKl:
    def _make_three_teachers(self):
        student = keras.random.normal((_B, _S, _V), seed=2)
        teacher_indices = []
        teacher_logits = []
        for t in range(_T):
            idx = ops.cast(
                ops.argsort(-keras.random.normal((_B, _S, _V), seed=10 + t), axis=-1)[..., :_K],
                "int32",
            )
            # Pluck student values at those indices for a "matched" teacher,
            # then add small per-teacher noise so they're distinct.
            vals = ops.take_along_axis(student, idx, axis=-1) + (t * 0.01)
            teacher_indices.append(idx)
            teacher_logits.append(vals)
        teacher_idx_stack = ops.stack(teacher_indices, axis=0)    # [T, B, S, K]
        teacher_log_stack = ops.stack(teacher_logits, axis=0)
        return student, teacher_log_stack, teacher_idx_stack

    def test_uniform_average_equals_mean_of_per_teacher_kls(self):
        """multi_teacher_kl_loss with no weights = unweighted mean."""
        student, t_logits, t_indices = self._make_three_teachers()
        per_teacher = []
        for t in range(_T):
            per_teacher.append(
                float(kl_div_topk_loss(student, t_logits[t], t_indices[t]))
            )
        expected = sum(per_teacher) / len(per_teacher)
        got = float(multi_teacher_kl_loss(student, t_logits, t_indices))
        assert got == pytest.approx(expected, rel=1.0e-5)

    def test_weighted_average_honors_weights(self):
        student, t_logits, t_indices = self._make_three_teachers()
        per_teacher = [
            float(kl_div_topk_loss(student, t_logits[t], t_indices[t]))
            for t in range(_T)
        ]
        weights = (1.0, 2.0, 4.0)
        weights_n = [w / sum(weights) for w in weights]
        expected = sum(w * k for w, k in zip(weights_n, per_teacher))
        got = float(
            multi_teacher_kl_loss(
                student, t_logits, t_indices, teacher_weights=weights
            )
        )
        assert got == pytest.approx(expected, rel=1.0e-5)


# --------------------------------------------------------------------------- #
# distillation_mixed_loss — α·CE + (1-α)·KL with fallback
# --------------------------------------------------------------------------- #
class TestDistillationMixed:
    def _setup(self):
        keras.utils.set_random_seed(13)
        student = keras.random.normal((_B, _S, _V), seed=13)
        labels = ops.cast(
            keras.random.uniform((_B, _S), maxval=_V, seed=14), "int32"
        )
        # Three teachers with random top-K positions AND independent
        # per-position noise on their logits — additive constants would
        # cancel under softmax, so we use multiplicative+noise to ensure
        # the teacher distribution actually differs from the student's.
        teacher_indices = []
        teacher_logits = []
        for t in range(_T):
            idx = ops.cast(
                ops.argsort(-student, axis=-1)[..., :_K], "int32"
            )
            base_vals = ops.take_along_axis(student, idx, axis=-1)
            noise = keras.random.normal(ops.shape(base_vals), seed=100 + t)
            vals = base_vals * 1.5 + noise  # scaled + noisy → different distribution
            teacher_indices.append(idx)
            teacher_logits.append(vals)
        return (
            student,
            labels,
            ops.stack(teacher_logits, axis=0),
            ops.stack(teacher_indices, axis=0),
        )

    def test_no_teacher_collapses_to_cross_entropy(self):
        student, labels, _, _ = self._setup()
        ce_total, ce_metrics = cross_entropy_with_z_loss(student, labels)
        total, metrics = distillation_mixed_loss(
            student, labels,
            teacher_topk_logits_per_teacher=None,
            teacher_topk_indices_per_teacher=None,
        )
        assert float(total) == pytest.approx(float(ce_total), rel=1.0e-5)
        assert float(metrics["kl"]) == 0.0
        assert float(metrics["alpha"]) == 1.0
        assert float(metrics["ce"]) == pytest.approx(float(ce_metrics["ce"]), rel=1.0e-5)

    def test_alpha_one_ignores_teacher(self):
        """α = 1.0 → pure CE, KL term has zero coefficient (but is computed)."""
        student, labels, t_logits, t_indices = self._setup()
        ce_total, _ = cross_entropy_with_z_loss(student, labels)
        total, metrics = distillation_mixed_loss(
            student, labels,
            t_logits, t_indices,
            alpha=1.0,
        )
        assert float(total) == pytest.approx(float(ce_total), rel=1.0e-5)
        # KL is still recorded for observability.
        assert float(metrics["kl"]) > 0.0

    def test_alpha_zero_is_pure_distillation(self):
        """α = 0.0 → pure KL, CE has zero coefficient."""
        student, labels, t_logits, t_indices = self._setup()
        kl_only = float(
            multi_teacher_kl_loss(student, t_logits, t_indices)
        )
        total, _ = distillation_mixed_loss(
            student, labels,
            t_logits, t_indices,
            alpha=0.0,
        )
        assert float(total) == pytest.approx(kl_only, rel=1.0e-5)

    def test_locked_alpha_0p3_lies_between_ce_and_kl(self):
        """At our locked α=0.3, total ≈ 0.3·CE + 0.7·KL."""
        student, labels, t_logits, t_indices = self._setup()
        ce_total, _ = cross_entropy_with_z_loss(student, labels)
        kl_only = multi_teacher_kl_loss(student, t_logits, t_indices)
        expected = 0.3 * float(ce_total) + 0.7 * float(kl_only)
        total, _ = distillation_mixed_loss(
            student, labels,
            t_logits, t_indices,
            alpha=0.3,
        )
        assert float(total) == pytest.approx(expected, rel=1.0e-5)

    def test_total_is_finite_and_positive(self):
        """Sanity: at α=0.3 the combined loss is a finite positive number."""
        student, labels, t_logits, t_indices = self._setup()
        total, metrics = distillation_mixed_loss(
            student, labels,
            t_logits, t_indices,
            alpha=0.3,
        )
        assert math.isfinite(float(total))
        assert float(total) > 0.0
        # All metric keys present.
        assert set(metrics.keys()) == {"ce", "z_loss", "kl", "alpha"}
