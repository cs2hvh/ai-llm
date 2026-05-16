"""Initial-state construction helpers (model + optimizer + state pytree).

These live here (not in ``scripts/run_pretrain.py``) so any library code —
notably ``myllm.infer.predict`` — can import them without the sys.path
gymnastics scripts require. The launcher in scripts/ still imports these,
so anything that depended on ``from scripts.run_pretrain import ...``
keeps working via the re-export at the bottom of that file.

Round B4 + post-Round-B refactor (2026-05-16): originally these were
defined directly in ``scripts/run_pretrain.py``. ``myllm.infer.predict``
hit a ``ModuleNotFoundError: No module named 'scripts'`` on the C1 pod
because ``src/myllm/`` is on sys.path (installed via ``pip -e``) but
``scripts/`` is not. Rather than maintain the sys.path hack we used as
a quick fix, library code now imports from this module.
"""
from __future__ import annotations

from typing import Any

from myllm.training.optimizer import (
    OptimizerConfig,
    build_optimizer,
    label_model_variables,
)
from myllm.utils import get_logger

log = get_logger(__name__)


def resolve_wsd_schedule_params(
    peak_lr: float,
    total_steps: int,
    *,
    lr_schedule_cfg: dict | None = None,
) -> dict:
    """Resolve WSD schedule parameters from yaml + sensible defaults.

    Pure function — testable without JAX/Keras. Returns dict with keys:
    ``warmup_steps``, ``decay_steps``, ``stable_steps``, ``end_lr``.

    P0 audit (2026-05-12): this used to be hardcoded inline in
    ``init_model_and_optimizer``, silently overriding yaml lr_schedule.
    Factored out so it can be unit-tested.

    Defaults when fields are absent:
        warmup_steps = min(2000, total_steps // 10)
        decay_fraction = 0.15
        end_lr_ratio = 0.1
    """
    schedule_cfg = lr_schedule_cfg or {}
    default_warmup = max(1, min(2000, total_steps // 10))
    warmup_steps = int(schedule_cfg.get("warmup_steps", default_warmup))
    decay_fraction = float(schedule_cfg.get("decay_fraction", 0.15))
    end_lr_ratio = float(schedule_cfg.get("end_lr_ratio", 0.1))
    decay_steps = max(1, int(total_steps * decay_fraction))
    stable_steps = max(0, total_steps - warmup_steps - decay_steps)
    end_lr = peak_lr * end_lr_ratio
    return {
        "warmup_steps": warmup_steps,
        "decay_steps": decay_steps,
        "stable_steps": stable_steps,
        "end_lr": end_lr,
    }


def init_model_and_optimizer(
    model_cfg: Any,
    opt_cfg: OptimizerConfig,
    total_steps: int,
    *,
    lr_schedule_cfg: dict | None = None,
):
    """Construct model, build it (allocate weights), wire optimizer.

    ``lr_schedule_cfg`` is the optional yaml `lr_schedule` block. When set,
    its fields override the hardcoded WSD defaults. P0 audit (2026-05-12)
    flagged that this used to be ignored — pilot's configured warmup/decay
    were silently overridden by hardcoded values.
    """
    from myllm.model.transformer import build_model

    log.info(
        "building_model",
        name=model_cfg.name,
        params_estimate=model_cfg.param_count_estimate(),
    )
    model = build_model(model_cfg)

    # Warmup-Stable-Decay schedule (playbook recommendation): linear warmup,
    # then constant peak_lr through the stable phase, then linear decay over
    # the last fraction. Doesn't commit to total_steps upfront — any stable-
    # phase checkpoint can be cooled in 10-15% of remaining compute.
    import optax

    sched = resolve_wsd_schedule_params(
        opt_cfg.peak_lr, total_steps, lr_schedule_cfg=lr_schedule_cfg
    )
    warmup_steps = sched["warmup_steps"]
    decay_steps = sched["decay_steps"]
    stable_steps = sched["stable_steps"]
    end_lr = sched["end_lr"]
    log.info(
        "lr_schedule_resolved",
        warmup_steps=warmup_steps,
        stable_steps=stable_steps,
        decay_steps=decay_steps,
        peak_lr=opt_cfg.peak_lr,
        end_lr=end_lr,
        source="yaml lr_schedule" if lr_schedule_cfg else "hardcoded defaults",
    )

    lr_fn = optax.join_schedules(
        schedules=[
            optax.linear_schedule(
                init_value=0.0,
                end_value=opt_cfg.peak_lr,
                transition_steps=warmup_steps,
            ),
            optax.constant_schedule(value=opt_cfg.peak_lr),
            optax.linear_schedule(
                init_value=opt_cfg.peak_lr,
                end_value=end_lr,
                transition_steps=decay_steps,
            ),
        ],
        boundaries=[warmup_steps, warmup_steps + stable_steps],
    )

    # muP per-parameter LR scaling. When model_cfg.mup is None, width_mult
    # collapses to 1.0 and build_optimizer returns the legacy single-AdamW
    # chain (no behavior change). When muP is set, hidden weights get LR
    # scaled by 1/width_mult.
    width_mult = model_cfg.mup_width_multiplier()
    param_labels = (
        label_model_variables(model) if model_cfg.mup is not None else None
    )
    if model_cfg.mup is not None:
        log.info(
            "mup_optimizer_active",
            width_mult=width_mult,
            base_width=model_cfg.mup.base_width,
            hidden_dim=model_cfg.hidden_dim,
            n_embedding=param_labels.count("embedding"),
            n_norm=param_labels.count("norm"),
            n_hidden=param_labels.count("hidden"),
        )

    optimizer = build_optimizer(
        opt_cfg, lr_fn,
        param_labels=param_labels,
        mup_width_mult=width_mult,
    )
    return model, optimizer


def initial_train_state(model, optimizer):
    """Construct the initial state dict consumed by ``loop.run``.

    Schema (also documented in ``myllm.training.loop._PERSIST_KEYS``):
      trainable_variables, non_trainable_variables, opt_state, step,
      lr_recovery_multiplier, data_position.

    2026-05-12 re-audit fix: data_position MUST be in the initial state
    so train_step's "preserve unknown keys" path has it from step 0.
    Without it, the loop's first state.get("data_position", 0) was always
    starting at 0 but never persisted into the train_step's new_state
    output — broke the checkpoint round-trip in subtle ways.
    """
    import jax.numpy as jnp

    trainable = [v.value for v in model.trainable_variables]
    non_trainable = [v.value for v in model.non_trainable_variables]
    opt_state = optimizer.init(trainable)
    return {
        "trainable_variables": trainable,
        "non_trainable_variables": non_trainable,
        "opt_state": opt_state,
        "step": 0,
        "lr_recovery_multiplier": jnp.float32(1.0),
        "data_position": 0,
    }
