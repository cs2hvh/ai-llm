"""Regression test for atomic NaN-skip in train_step.

2026-05-12 audit caught a real bug: the prior NaN-skip implementation
zeroed gradients but still called ``optimizer.update`` + ``apply_updates``.
With AdamW (weight_decay > 0), this still applies ``lr * wd * params`` to
the parameters on every "skipped" step, AND the optimizer's internal step
counter advances inside ``update()``. So "skipped" batches silently drifted
params and corrupted bias-correction denominators.

The fix is atomic: build the candidate update assuming the batch is good,
then ``jnp.where(step_ok, candidate, old)`` on every leaf of (params,
opt_state, non_trainable) so the entire state reverts when the batch was
bad.

This file does NOT depend on keras (which is currently skipped in CI due
to the keras+tensorflow detection); we test the atomic-where pattern
directly against a tiny optax AdamW chain.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
optax = pytest.importorskip("optax")


def _atomic_nan_skip_update(params, opt_state, grads, optimizer, mult=1.0):
    """The atomic-where pattern from train_step.py.

    Returns (new_params, new_opt_state, step_ok). When step_ok is False,
    new_params == params and new_opt_state == opt_state (no drift).
    """
    grads_finite = jax.tree.reduce(
        lambda a, b: a & b,
        jax.tree.map(lambda g: jnp.all(jnp.isfinite(g)), grads),
        jnp.array(True),
    )
    step_ok = grads_finite

    updates, candidate_opt_state = optimizer.update(grads, opt_state, params)
    updates = jax.tree.map(lambda u: u * mult, updates)
    candidate_params = optax.apply_updates(params, updates)

    def _pick(new_leaf, old_leaf):
        return jnp.where(step_ok, new_leaf, old_leaf)

    new_params = jax.tree.map(_pick, candidate_params, params)
    new_opt_state = jax.tree.map(_pick, candidate_opt_state, opt_state)
    return new_params, new_opt_state, step_ok


def _zero_grads_skip_update(params, opt_state, grads, optimizer, mult=1.0):
    """The OLD (broken) NaN-skip from the 2026-05-11 version of train_step.

    Just zeroes grads on bad-batch steps. With AdamW weight_decay > 0,
    this is NOT a no-op — weight decay still applies, bias correction
    still advances.
    """
    grads_finite = jax.tree.reduce(
        lambda a, b: a & b,
        jax.tree.map(lambda g: jnp.all(jnp.isfinite(g)), grads),
        jnp.array(True),
    )
    safe_grads = jax.tree.map(
        lambda g: jnp.where(grads_finite, g, jnp.zeros_like(g)),
        grads,
    )
    updates, new_opt_state = optimizer.update(safe_grads, opt_state, params)
    updates = jax.tree.map(lambda u: u * mult, updates)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, grads_finite


def _make_tiny_state(weight_decay=0.1, lr=1e-2):
    """Tiny params + AdamW state pre-warmed by 5 good steps."""
    params = {"w": jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)}
    optimizer = optax.adamw(learning_rate=lr, weight_decay=weight_decay, b1=0.9, b2=0.95)
    opt_state = optimizer.init(params)
    # Warm up moments with 5 normal gradient steps so m, v are non-zero.
    for _ in range(5):
        grads = {"w": jnp.array([0.1, 0.1, 0.1, 0.1], dtype=jnp.float32)}
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
    return params, opt_state, optimizer


# --------------------------------------------------------------------------- #
# The actual regression: atomic NaN-skip preserves state exactly.
# --------------------------------------------------------------------------- #
class TestAtomicNanSkip:
    def test_atomic_skip_preserves_params_exactly(self):
        """With NaN grads, params must be byte-identical to before."""
        params, opt_state, optimizer = _make_tiny_state(weight_decay=0.1)

        snapshot_w = np.array(params["w"])
        nan_grads = {"w": jnp.array([jnp.nan, 0.1, 0.1, 0.1], dtype=jnp.float32)}

        new_params, _, step_ok = _atomic_nan_skip_update(
            params, opt_state, nan_grads, optimizer
        )

        assert not step_ok, "step_ok should be False when grads have NaN"
        np.testing.assert_array_equal(
            np.array(new_params["w"]), snapshot_w,
            err_msg="atomic NaN-skip must leave params EXACTLY unchanged. "
                    "If this fails, AdamW weight decay is leaking through "
                    "the skip — see train_step.py P0-1 fix."
        )

    def test_atomic_skip_preserves_optimizer_moments(self):
        """The Adam m and v moments must be byte-identical to before."""
        params, opt_state, optimizer = _make_tiny_state()
        m_before = jax.tree.leaves(opt_state)

        nan_grads = {"w": jnp.array([jnp.inf, 0.0, 0.0, 0.0], dtype=jnp.float32)}
        _, new_opt_state, step_ok = _atomic_nan_skip_update(
            params, opt_state, nan_grads, optimizer
        )

        assert not step_ok
        m_after = jax.tree.leaves(new_opt_state)
        assert len(m_before) == len(m_after)
        for b, a in zip(m_before, m_after):
            np.testing.assert_array_equal(
                np.array(a), np.array(b),
                err_msg="optimizer state (m, v, step counter) must be EXACTLY "
                        "unchanged on a skipped batch — bias correction depends "
                        "on the step count not advancing for bad batches."
            )

    def test_atomic_skip_lets_good_batches_through(self):
        """Sanity: a finite gradient still produces a real update."""
        params, opt_state, optimizer = _make_tiny_state()
        before_w = np.array(params["w"])

        good_grads = {"w": jnp.array([0.5, 0.5, 0.5, 0.5], dtype=jnp.float32)}
        new_params, _, step_ok = _atomic_nan_skip_update(
            params, opt_state, good_grads, optimizer
        )

        assert bool(step_ok), "step_ok should be True for finite grads"
        # Params should have changed.
        assert not np.allclose(np.array(new_params["w"]), before_w)


# --------------------------------------------------------------------------- #
# Demonstration: the OLD zero-grads approach was actually broken.
# This test is here to PROVE the bug we fixed was real, not theoretical.
# --------------------------------------------------------------------------- #
class TestOldZeroGradsApproachWasBroken:
    def test_zero_grads_still_drifts_params_via_weight_decay(self):
        """The pre-2026-05-12 code zeroed grads but still called
        optimizer.update + apply_updates. With AdamW weight_decay=0.1,
        params get a `-lr * wd * params` push every "skipped" step.

        For a param value of 4.0 with lr=1e-2, wd=0.1, the per-step drift
        is roughly 1e-2 * 0.1 * 4.0 = 4e-3. Easily detectable after one
        "skipped" batch.

        If this test FAILS (i.e. params don't drift), it means optax's
        AdamW behavior changed and we should re-derive the safety
        argument. If it PASSES, our fix is necessary."""
        params, opt_state, optimizer = _make_tiny_state(weight_decay=0.1, lr=1e-2)
        snapshot_w = np.array(params["w"])

        nan_grads = {"w": jnp.array([jnp.nan, 0.0, 0.0, 0.0], dtype=jnp.float32)}
        old_new_params, _, _ = _zero_grads_skip_update(
            params, opt_state, nan_grads, optimizer
        )

        # Params SHOULD have drifted under the old code (this proves
        # the old code was broken).
        old_w = np.array(old_new_params["w"])
        max_drift = float(np.max(np.abs(old_w - snapshot_w)))
        assert max_drift > 1e-4, (
            f"old zero-grads approach should drift params via weight decay; "
            f"observed max drift = {max_drift}. If this is 0, AdamW's wd "
            f"behavior changed and our P0-1 fix may no longer be needed."
        )


# --------------------------------------------------------------------------- #
# Bias-correction step counter: another subtle drift the old code allowed.
# --------------------------------------------------------------------------- #
class TestStepCounterPreservation:
    def test_old_approach_advances_internal_step_counter(self):
        """AdamW's bias correction uses `1 - β^t` where t is the optimizer's
        internal step counter (separate from our `state["step"]`). If
        skipped batches advance t, then the bias-correction denominators
        drift — silently changing the effective LR over time."""
        params, opt_state, optimizer = _make_tiny_state()

        # Find the step counter leaf in opt_state (it's inside ScaleByAdamState).
        def _find_count(tree):
            for leaf in jax.tree.leaves(tree):
                arr = np.asarray(leaf)
                if arr.dtype.kind == "i" and arr.shape == ():
                    return int(arr)
            return None

        count_before = _find_count(opt_state)
        nan_grads = {"w": jnp.array([jnp.nan, 0.0, 0.0, 0.0], dtype=jnp.float32)}
        _, old_opt_state, _ = _zero_grads_skip_update(
            params, opt_state, nan_grads, optimizer
        )
        count_after_old = _find_count(old_opt_state)
        assert count_after_old == count_before + 1, (
            "Old zero-grads approach advances optimizer step counter "
            "(see P0-1 audit). This is the silent-drift bug we fixed."
        )

    def test_atomic_approach_preserves_step_counter(self):
        params, opt_state, optimizer = _make_tiny_state()

        def _find_count(tree):
            for leaf in jax.tree.leaves(tree):
                arr = np.asarray(leaf)
                if arr.dtype.kind == "i" and arr.shape == ():
                    return int(arr)
            return None

        count_before = _find_count(opt_state)
        nan_grads = {"w": jnp.array([jnp.nan, 0.0, 0.0, 0.0], dtype=jnp.float32)}
        _, new_opt_state, _ = _atomic_nan_skip_update(
            params, opt_state, nan_grads, optimizer
        )
        count_after = _find_count(new_opt_state)
        assert count_after == count_before, (
            f"atomic NaN-skip must preserve the optimizer step counter; "
            f"was {count_before}, became {count_after}"
        )
