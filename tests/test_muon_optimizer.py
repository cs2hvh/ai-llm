"""Muon hybrid optimizer regression tests — D10 (post-review 2026-05-18).

Validates the Muon hybrid path added to ``build_optimizer``:

  1. ``use_muon=False`` (default) → identical behavior to the
     pre-D10 AdamW path. Backwards-compat is locked.

  2. ``use_muon=True`` with the muP 3-bucket label split → the hidden
     bucket routes through ``optax.contrib.muon``; embedding + norm
     stay on AdamW. Single-step finite + nonzero update magnitudes.

  3. ``use_muon=True`` + ``mup_width_mult > 1`` → hidden bucket still
     gets the 1/width_mult LR scaling on top of Muon (we documented
     that the "Muon makes muP redundant" claim is DISPUTED; we keep
     the scaling as the conservative default until empirically
     refuted).

  4. Optimizer state is Orbax-serialisable (pytree of arrays — no
     opaque types).

Tests run on CPU; no model required. We use the same toy-param /
toy-grad fixtures as ``test_mup_optim.py`` so the assertions speak
the same language.
"""
from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
optax = pytest.importorskip("optax")

# Skip the entire module on environments where optax.contrib.muon
# isn't shipped yet. We pin optax>=0.2.5 in requirements; this guard
# is for developer envs that lag.
if not hasattr(optax, "contrib") or not hasattr(optax.contrib, "muon"):
    pytest.skip(
        "optax.contrib.muon not available (need optax>=0.2.5)",
        allow_module_level=True,
    )

from myllm.training.optimizer import (  # noqa: E402
    PARAM_GROUP_EMBEDDING,
    PARAM_GROUP_HIDDEN,
    PARAM_GROUP_NORM,
    OptimizerConfig,
    build_optimizer,
)


def _const_lr(_step):
    return 1.0e-3


def _toy_params():
    # Same shapes as test_mup_optim.py for direct cross-referencing.
    # Hidden weight is square so Muon's Newton-Schulz is well-defined.
    return [
        jnp.zeros((4, 8)),   # embedding (>=2D — Muon's bucket if not routed away)
        jnp.zeros((4,)),     # norm (1D — AdamW always)
        jnp.zeros((4, 4)),   # hidden (square 2D — Muon when use_muon=True)
    ]


def _toy_grads():
    return [
        jnp.ones((4, 8)),
        jnp.ones((4,)),
        jnp.ones((4, 4)),
    ]


# --------------------------------------------------------------------------- #
# Backwards compatibility: use_muon=False matches pre-D10 optimizer
# --------------------------------------------------------------------------- #
def test_use_muon_false_matches_legacy_adamw_path():
    """OptimizerConfig(use_muon=False) is the default; the optimizer it
    builds must be IDENTICAL in behavior to the pre-D10 single-AdamW
    chain (no Muon imports touched on this path)."""
    cfg = OptimizerConfig(peak_lr=1.0e-3, weight_decay=0.0, use_muon=False)
    opt = build_optimizer(cfg, _const_lr)

    params = _toy_params()
    opt_state = opt.init(params)
    updates, _ = opt.update(_toy_grads(), opt_state, params)

    # All three updates have the same magnitude (legacy single-AdamW
    # behavior — same assertion as test_default_off_matches_legacy_optimizer).
    mags = [float(jnp.mean(jnp.abs(u))) for u in updates]
    assert max(mags) - min(mags) < 1.0e-6, (
        f"use_muon=False should match legacy AdamW; got per-group {mags}"
    )


# --------------------------------------------------------------------------- #
# Muon hybrid path — hidden goes through Muon, embedding + norm through AdamW
# --------------------------------------------------------------------------- #
def test_muon_hybrid_produces_finite_updates_at_single_step():
    """build_optimizer with use_muon=True must produce finite, nonzero
    updates on the first step for all 3 buckets.

    We don't assert specific magnitudes — Muon's spectral-norm bound
    + Newton-Schulz iteration produce updates that aren't directly
    comparable to AdamW's coordinate-wise pre-conditioning. We just
    require finite and nonzero.
    """
    cfg = OptimizerConfig(peak_lr=1.0e-3, weight_decay=0.0, use_muon=True)
    labels = [PARAM_GROUP_EMBEDDING, PARAM_GROUP_NORM, PARAM_GROUP_HIDDEN]
    opt = build_optimizer(cfg, _const_lr, param_labels=labels, mup_width_mult=1.0)

    params = _toy_params()
    opt_state = opt.init(params)
    updates, _ = opt.update(_toy_grads(), opt_state, params)

    for i, u in enumerate(updates):
        u_np = jax.device_get(u)
        assert jnp.all(jnp.isfinite(u_np)), f"bucket {i} produced non-finite updates"
        assert float(jnp.mean(jnp.abs(u_np))) > 0.0, (
            f"bucket {i} produced zero updates (optimizer stuck?)"
        )


def test_muon_hybrid_width_mult_scales_hidden_bucket():
    """At width_mult=2, the hidden bucket's update magnitude (Muon)
    must be reduced relative to width_mult=1. We don't compare absolute
    magnitudes across embedding/hidden (different optimizers), only
    width_mult=1 vs width_mult=2 within the hidden bucket.
    """
    cfg = OptimizerConfig(peak_lr=1.0e-3, weight_decay=0.0, use_muon=True)
    labels = [PARAM_GROUP_EMBEDDING, PARAM_GROUP_NORM, PARAM_GROUP_HIDDEN]

    opt_w1 = build_optimizer(cfg, _const_lr, param_labels=labels, mup_width_mult=1.0)
    opt_w2 = build_optimizer(cfg, _const_lr, param_labels=labels, mup_width_mult=2.0)

    params = _toy_params()
    upd_w1, _ = opt_w1.update(_toy_grads(), opt_w1.init(params), params)
    upd_w2, _ = opt_w2.update(_toy_grads(), opt_w2.init(params), params)

    # Hidden bucket (index 2) — width_mult=2 should give ~half the
    # update magnitude (the 1/width_mult scale is applied
    # post-Muon, multiplicatively, regardless of Muon's internal
    # spectral-norm bound).
    h_w1 = float(jnp.mean(jnp.abs(upd_w1[2])))
    h_w2 = float(jnp.mean(jnp.abs(upd_w2[2])))
    ratio = h_w2 / h_w1
    assert ratio == pytest.approx(0.5, rel=5.0e-3), (
        f"hidden bucket under Muon should scale 1/width_mult; got ratio {ratio:.6f}"
    )

    # Embedding bucket (index 0) — width_mult shouldn't affect it
    # (still on AdamW with LR scale 1.0).
    e_w1 = float(jnp.mean(jnp.abs(upd_w1[0])))
    e_w2 = float(jnp.mean(jnp.abs(upd_w2[0])))
    assert abs(e_w1 - e_w2) < 1.0e-7, (
        f"embedding bucket should be width-mult-invariant; "
        f"got {e_w1} vs {e_w2}"
    )


def test_muon_optimizer_state_is_orbax_serialisable_pytree():
    """The optimizer state must be a plain pytree of arrays — no opaque
    types — so Orbax can round-trip it. We check by walking the leaves
    and confirming every leaf is a JAX array (or scalar)."""
    cfg = OptimizerConfig(peak_lr=1.0e-3, weight_decay=0.0, use_muon=True)
    labels = [PARAM_GROUP_EMBEDDING, PARAM_GROUP_NORM, PARAM_GROUP_HIDDEN]
    opt = build_optimizer(cfg, _const_lr, param_labels=labels, mup_width_mult=1.0)
    state = opt.init(_toy_params())

    leaves = jax.tree_util.tree_leaves(state)
    assert len(leaves) > 0, "optimizer state should have at least one leaf"
    for leaf in leaves:
        # Either a JAX array or a tracer-friendly scalar/array-likable.
        assert hasattr(leaf, "shape") or isinstance(leaf, (int, float)), (
            f"opt state leaf has unexpected type {type(leaf)} — not Orbax-friendly"
        )
