"""Regression test for B1 (2026-05-12 audit): Orbax must round-trip
``optax.MultiTransformState`` namedtuples through save/restore so the
muP optimizer state survives checkpointing.

Before this fix, ``CheckpointManager.restore()`` called Orbax without a
template, so the restored ``opt_state`` came back as a plain dict instead
of a ``MultiTransformState`` namedtuple. Subsequent ``optimizer.update()``
calls would fail with ``AttributeError: 'dict' object has no attribute
'inner_states'``.

The fix is to pass the live ``optimizer.init(...)`` output as a template
to Orbax's restore, which tells it to rebuild the namedtuple structure.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
optax = pytest.importorskip("optax")

from myllm.training.checkpoint import CheckpointConfig, CheckpointManager


def _build_multi_transform_optimizer():
    """Build a small optax multi_transform optimizer with two groups
    (mimicking the muP `embedding` vs `hidden` split)."""
    # Two parameter groups: "emb" gets full LR, "hidden" gets scaled LR.
    optimizers = {
        "emb": optax.adamw(learning_rate=1e-3, weight_decay=0.0, b1=0.9, b2=0.95),
        "hidden": optax.adamw(learning_rate=1e-3 / 4.0, weight_decay=0.0, b1=0.9, b2=0.95),
    }
    # multi_transform expects a function that classifies each leaf by label.
    # We'll structure params as a dict and label each key.
    optimizer = optax.multi_transform(
        optimizers,
        param_labels={"w_emb": "emb", "w_hidden": "hidden"},
    )
    return optimizer


def _make_state():
    """Build a state pytree that mirrors what loop.py persists."""
    params = {
        "w_emb": jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32),
        "w_hidden": jnp.array([0.5, 1.5, 2.5], dtype=jnp.float32),
    }
    optimizer = _build_multi_transform_optimizer()
    opt_state = optimizer.init(params)

    # Take 2 update steps so opt_state moments are non-zero (proves the
    # template handles populated namedtuples, not just empty defaults).
    grads = {
        "w_emb": jnp.array([0.01, 0.01, 0.01], dtype=jnp.float32),
        "w_hidden": jnp.array([0.02, 0.02, 0.02], dtype=jnp.float32),
    }
    for _ in range(2):
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

    return {
        "trainable_variables": params,
        "non_trainable_variables": [],
        "opt_state": opt_state,
        "step": 2,
        "lr_recovery_multiplier": jnp.float32(1.0),
        "data_position": 64,
    }, optimizer


# --------------------------------------------------------------------------- #
# The actual P0-1 regression
# --------------------------------------------------------------------------- #
def test_multi_transform_state_type_preserved_with_template(tmp_path):
    """save → restore(template=...) must return a state whose opt_state is
    structurally identical to the original (specifically, a
    MultiTransformState namedtuple, not a plain dict)."""
    state, _optimizer = _make_state()
    ckpt = CheckpointManager(CheckpointConfig(root=str(tmp_path / "ckpts"),
                                              keep_last_n=1, keep_every_n=10000))
    ckpt.save(2, state)

    # Build a fresh template (as the loop would: fresh state from
    # init_train_state, opt_state from optimizer.init() with the same
    # optimizer factory).
    template_state, _ = _make_state()  # same shapes, fresh moments
    template = {k: template_state[k] for k in (
        "trainable_variables", "non_trainable_variables", "opt_state",
        "step", "lr_recovery_multiplier", "data_position"
    )}

    restored = ckpt.restore(2, template=template)

    # Critical: opt_state must NOT be a plain dict. The MultiTransformState
    # namedtuple has an `inner_states` attribute that optax.update() reads.
    assert hasattr(restored["opt_state"], "inner_states"), (
        "B1 regression: restored opt_state is missing `inner_states`. "
        "Either it's a plain dict (Orbax restore-without-template path) or "
        "the template wasn't matched. Without `inner_states`, the next "
        "optimizer.update() will AttributeError."
    )

    # The restored opt_state should be a working optax state — try an
    # update step.
    _, restored_optimizer = _make_state()
    grads = {
        "w_emb": jnp.array([0.01, 0.01, 0.01], dtype=jnp.float32),
        "w_hidden": jnp.array([0.02, 0.02, 0.02], dtype=jnp.float32),
    }
    # This is the call that would fail under the old (no-template) restore.
    updates, new_opt_state = restored_optimizer.update(
        grads, restored["opt_state"], restored["trainable_variables"]
    )
    # If we got here, the update succeeded. Sanity-check the updates aren't
    # NaN/Inf.
    for leaf in jax.tree.leaves(updates):
        arr = np.asarray(leaf)
        assert np.all(np.isfinite(arr)), "updates contain non-finite values"


def test_restore_without_template_still_works_for_simple_states(tmp_path):
    """Back-compat: states without namedtuples should still restore via
    the no-template path (this is what tests of the watchdog/data-cursor
    flow do — they don't have a real optimizer)."""
    state = {
        "trainable_variables": [jnp.array([1.0, 2.0, 3.0])],
        "non_trainable_variables": [],
        "opt_state": [jnp.array([0.0, 0.0, 0.0])],  # plain list, no namedtuple
        "step": 5,
        "lr_recovery_multiplier": jnp.float32(1.0),
        "data_position": 100,
    }
    ckpt = CheckpointManager(CheckpointConfig(root=str(tmp_path / "ckpts"),
                                              keep_last_n=1, keep_every_n=10000))
    ckpt.save(5, state)
    # No template → returns plain pytree (no namedtuples to reconstruct).
    restored = ckpt.restore(5)
    assert int(restored["step"]) == 5
    assert int(restored["data_position"]) == 100


def test_opt_state_moments_byte_identical_after_roundtrip(tmp_path):
    """Values must be byte-identical (or near-identical for fp32) across
    save/restore. If the namedtuple roundtrip silently re-initializes
    moments, training would lose the bias-correction history."""
    state, optimizer = _make_state()
    ckpt = CheckpointManager(CheckpointConfig(root=str(tmp_path / "ckpts"),
                                              keep_last_n=1, keep_every_n=10000))
    ckpt.save(2, state)

    template_state, _ = _make_state()
    template = {k: template_state[k] for k in (
        "trainable_variables", "non_trainable_variables", "opt_state",
        "step", "lr_recovery_multiplier", "data_position"
    )}
    restored = ckpt.restore(2, template=template)

    # Compare every leaf in the original vs restored opt_state.
    orig_leaves = jax.tree.leaves(state["opt_state"])
    restored_leaves = jax.tree.leaves(restored["opt_state"])
    assert len(orig_leaves) == len(restored_leaves), (
        f"leaf count mismatch: orig {len(orig_leaves)} vs restored {len(restored_leaves)}"
    )
    for o, r in zip(orig_leaves, restored_leaves):
        np.testing.assert_array_equal(
            np.asarray(o), np.asarray(r),
            err_msg="opt_state leaf differs after Orbax roundtrip — moments lost?"
        )
