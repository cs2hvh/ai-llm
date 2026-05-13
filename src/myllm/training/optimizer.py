"""Optax optimizer factory.

We chain in this order:
    1. ``clip_by_global_norm`` — bound gradient magnitude pre-Adam.
    2. ``adamw`` — the workhorse optimizer.
       (With muP enabled, this becomes a ``multi_transform`` so that
        different parameter groups can have different effective LRs.)

This is a pure factory; no JAX state is materialised here.

R1 muP support (2026-05-11):
    When a ``MupConfig`` is attached to the model, hidden-weight params
    (Q/K/V/O/gate/up/down) get LR scaled by ``1 / width_mult``, while
    embedding and norm params keep the base LR. See ``docs/mup_design.md``
    §"Change 4" for the recipe. When muP is disabled (``mup_width_mult=1``),
    the optimizer collapses back to a single AdamW with no per-group split.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OptimizerConfig:
    peak_lr: float = 2.0e-4
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.1
    eps: float = 1.0e-8
    grad_clip_global_norm: float = 1.0


# Param-group labels used by the muP multi-transform optimizer.
# Each trainable variable falls into exactly one group:
#   embedding — token embedding + (untied) lm_head: keep base LR
#   norm      — RMSNorm scales (incl. q_norm / k_norm): keep base LR
#   hidden    — all weight matrices in attention + FFN: LR × (1 / width_mult)
PARAM_GROUP_EMBEDDING = "embedding"
PARAM_GROUP_NORM = "norm"
PARAM_GROUP_HIDDEN = "hidden"
_PARAM_GROUPS = (PARAM_GROUP_EMBEDDING, PARAM_GROUP_NORM, PARAM_GROUP_HIDDEN)


def label_variable_for_mup(path: str) -> str:
    """Classify a single variable by its Keras path string into one of
    ``embedding`` / ``norm`` / ``hidden``.

    The classification is purely path-based; we look for substrings:
        - ``embed`` (matches ``tok_embed``)         → embedding
        - ``lm_head`` (untied LM head)              → embedding
        - ``norm`` (matches attn_norm/ffn_norm/q_norm/k_norm/final_norm) → norm
        - anything else (wq/wk/wv/wo/w_gate/w_up/w_down)                  → hidden

    This is intentionally simple and predictable so unit tests can verify
    classification without hardcoding model structure. If new layer kinds
    are added, this function must be revisited.
    """
    p = (path or "").lower()
    if "embed" in p:
        return PARAM_GROUP_EMBEDDING
    if "lm_head" in p:
        return PARAM_GROUP_EMBEDDING
    if "norm" in p:
        return PARAM_GROUP_NORM
    return PARAM_GROUP_HIDDEN


def label_model_variables(model: Any) -> list[str]:
    """Return one label per element of ``model.trainable_variables``.

    Same ordering as the model's variable list, so the labels can be used
    directly as the ``param_labels`` argument to ``optax.multi_transform``.
    """
    labels: list[str] = []
    for v in model.trainable_variables:
        path = getattr(v, "path", "") or getattr(v, "name", "") or ""
        labels.append(label_variable_for_mup(path))
    return labels


def build_optimizer(
    config: OptimizerConfig,
    lr_fn: Callable[[int], float],
    *,
    param_labels: list[str] | None = None,
    mup_width_mult: float = 1.0,
) -> Any:
    """Return a chained Optax GradientTransformation.

    Args:
        config:           optimizer hyperparameters (peak LR, betas, wd, eps,
                          grad-clip threshold).
        lr_fn:            ``step → lr`` callable, e.g. our WSD schedule.
        param_labels:     optional. When provided alongside ``mup_width_mult > 1.0``,
                          the optimizer applies different LR multipliers per
                          parameter group via ``optax.multi_transform``. Each
                          entry must be one of ``embedding`` / ``norm`` /
                          ``hidden``. Length must equal the number of trainable
                          variables. Use ``label_model_variables(model)`` to
                          build this.
        mup_width_mult:   the model's muP width multiplier (``hidden_dim /
                          mup.base_width``). When ``1.0`` (default) muP is
                          disabled and the optimizer collapses to a single
                          AdamW. When ``> 1.0``, hidden weights get LR
                          scaled by ``1 / mup_width_mult``.

    Backwards-compatibility: calling ``build_optimizer(config, lr_fn)``
    without the muP kwargs yields the same optimizer the project has used
    since day one.
    """
    try:
        import optax
    except ImportError as e:
        raise ImportError(
            "optax not installed; install with `pip install optax`"
        ) from e

    # 2026-05-12 senior review P0-2: pin Adam first moment to fp32
    # defensively.
    #
    # Optax 0.2.5's `optax.adamw` accepts `mu_dtype` (first moment) but
    # NOT `nu_dtype` (second moment, which defaults to param dtype via
    # `jnp.zeros_like(params)`).
    #
    # **Verified 2026-05-12**: under our Keras 3 mixed_precision="mixed_bfloat16"
    # policy, variables are stored as fp32 (compute happens in bf16 via
    # autocast). So `optax.init(params_fp32)` produces fp32 m AND fp32 nu
    # already — the reviewer's "weaker Adam second-moment precision" risk
    # does NOT manifest in our current setup.
    #
    # Pinning `mu_dtype=fp32` is defensive: it locks the first moment to
    # fp32 regardless of any future policy change (e.g. if someone moves
    # the Keras policy to true bf16-storage mixed precision). The second
    # moment will track param dtype until we upgrade Optax to a version
    # that exposes `nu_dtype`. Add a post-init dtype audit to run_pretrain
    # to catch policy drift early.
    import jax.numpy as jnp
    _FP32 = jnp.float32

    # Fast path: no muP, return the original single-AdamW chain.
    if mup_width_mult == 1.0 or param_labels is None:
        return optax.chain(
            optax.clip_by_global_norm(config.grad_clip_global_norm),
            optax.adamw(
                learning_rate=lr_fn,
                b1=config.beta1,
                b2=config.beta2,
                eps=config.eps,
                weight_decay=config.weight_decay,
                mu_dtype=_FP32,
            ),
        )

    # Validate the muP path inputs.
    if mup_width_mult <= 0:
        raise ValueError(f"mup_width_mult must be > 0, got {mup_width_mult}")
    bad = [lbl for lbl in param_labels if lbl not in _PARAM_GROUPS]
    if bad:
        raise ValueError(
            f"param_labels contains unknown groups {set(bad)}; "
            f"expected only {_PARAM_GROUPS}"
        )

    def _adamw_with_scale(lr_scale: float):
        """AdamW followed by a static per-group multiplier.

        Optax applies negative-LR convention internally (updates are
        subtracted from params), so multiplying the post-AdamW update
        by ``lr_scale`` rescales the effective LR by exactly ``lr_scale``.
        """
        return optax.chain(
            optax.adamw(
                learning_rate=lr_fn,
                b1=config.beta1,
                b2=config.beta2,
                eps=config.eps,
                weight_decay=config.weight_decay,
                mu_dtype=_FP32,    # P0-2 fix (2026-05-12) — see top of function
            ),
            optax.scale(lr_scale),
        )

    return optax.chain(
        optax.clip_by_global_norm(config.grad_clip_global_norm),
        optax.multi_transform(
            transforms={
                PARAM_GROUP_EMBEDDING: _adamw_with_scale(1.0),
                PARAM_GROUP_NORM: _adamw_with_scale(1.0),
                PARAM_GROUP_HIDDEN: _adamw_with_scale(1.0 / mup_width_mult),
            },
            param_labels=param_labels,
        ),
    )


# --------------------------------------------------------------------------- #
# FSDP / ZeRO-3 optimizer-state sharding (2026-05-13)
#
# Companion to `myllm.training.mesh.make_param_shardings`. Builds a
# NamedSharding pytree that matches the *shape* of `optimizer.init(params)`
# without actually allocating any opt-state. Used by run_pretrain.py's
# sharded-init flow:
#
#     opt_state_sharding = make_optimizer_state_sharding(
#         optimizer, trainable_sharded, mesh,
#     )
#     opt_init_jit = jax.jit(optimizer.init, out_shardings=opt_state_sharding)
#     opt_state = opt_init_jit(trainable_sharded)
#
# Why this is its own function (not just a call to make_param_shardings):
#   - We use `jax.eval_shape` to query the optimizer's output structure
#     without running it. That gives us a pytree of ShapeDtypeStruct in
#     EXACTLY the same nested form as the real opt_state — including
#     namedtuple types (optax.MultiTransformState, ScaleByAdamState, etc.).
#   - Critical: muP uses optax.multi_transform whose state IS a namedtuple
#     (`MultiTransformState`). If a sharding helper accidentally flattens
#     the tree (e.g. via tree_leaves -> tree_unflatten with a generic
#     treedef), restore via `template=` would still work BUT live
#     opt-state updates would break: `state.inner_states` (a namedtuple
#     field access) becomes a dict key access and silently fails. This
#     is the same bug class as the B1 fix from 2026-05-12 audit.
#   - Using jax.tree.map preserves namedtuple types by default. Verified
#     by the test_namedtuple_preserved test in tests/test_mesh_fsdp.py.
# --------------------------------------------------------------------------- #
def make_optimizer_state_sharding(
    optimizer: Any,
    params: Any,
    mesh: Any,
    mesh_axis: str = "data",
) -> Any:
    """Derive a NamedSharding pytree for ``optimizer.init(params)``.

    Args:
        optimizer: an ``optax.GradientTransformation`` (the return value of
            ``make_optimizer(...)``).
        params: the params pytree the optimizer will operate on. Can be
            real arrays or ``jax.ShapeDtypeStruct`` (eval_shape style).
            Only the *shape and dtype* of each leaf is used.
        mesh: the JAX ``Mesh`` (from ``build_mesh_and_shardings``).
        mesh_axis: the axis name to shard along (default ``"data"``).

    Returns:
        A PyTree of ``NamedSharding`` with the SAME tree structure as
        ``optimizer.init(params)`` would produce. Pass this as
        ``out_shardings`` to ``jax.jit(optimizer.init, ...)`` so the
        initial opt-state lands sharded without an unsharded transient.
    """
    try:
        import jax
    except ImportError as e:
        raise ImportError("jax not installed; install jax[cuda12]") from e
    from myllm.training.mesh import make_param_shardings

    # eval_shape returns a pytree of ShapeDtypeStruct in the SAME structure
    # as optimizer.init(params) — including namedtuples like
    # MultiTransformState / ScaleByAdamState. No real allocation occurs.
    abstract_state = jax.eval_shape(lambda p: optimizer.init(p), params)
    return make_param_shardings(abstract_state, mesh, mesh_axis=mesh_axis)
