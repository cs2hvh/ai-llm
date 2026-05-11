"""Pretraining loop with watchdog auto-rollback.

Owns the wall-clock orchestration of training:
    - read batches from the data pipeline
    - call the JIT'd ``train_step``
    - log to W&B + structlog
    - checkpoint on the configured cadence
    - run the watchdog and self-heal on hard spikes (rollback + halve LR + skip)
    - resume from the latest complete checkpoint at startup

Hard-spike recovery policy:
    1. Save a failure-trace marker checkpoint at the spike point.
    2. Restore the most recent *good* checkpoint (one strictly before the
       spike).
    3. Halve ``state["lr_recovery_multiplier"]`` so subsequent steps run at
       a lower effective LR.
    4. Skip ``recovery_skip_batches`` batches from the data iterator so the
       offending batch is not replayed.
    5. Reset the watchdog statistics (the post-restore loss is a new regime).
    6. Resume training.

After ``max_recoveries`` consecutive hard spikes, the loop gives up and
raises ``LossSpikeError`` — at that point human intervention is needed.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from myllm.training.checkpoint import CheckpointConfig, CheckpointManager
from myllm.training.watchdog import LossSpikeWatchdog
from myllm.utils import get_logger
from myllm.utils.exceptions import LossSpikeError, TrainingError

log = get_logger(__name__)


@dataclass(frozen=True)
class LoopConfig:
    total_steps: int
    log_every: int = 10
    checkpoint_every: int = 1000
    eval_every: int | None = None
    cost_ceiling_usd: float | None = None
    # Auto-rollback policy (defaults are conservative).
    recovery_skip_batches: int = 100
    recovery_lr_decay: float = 0.5
    max_recoveries: int = 3


# State keys we persist + restore. Updated to match the post-mesh state schema.
_PERSIST_KEYS = (
    "trainable_variables",
    "non_trainable_variables",
    "opt_state",
    "step",
    "lr_recovery_multiplier",
)


def run(
    train_step_fn: Any,
    initial_state: dict[str, Any],
    data_iter: Iterable[dict[str, Any]],
    *,
    loop_config: LoopConfig,
    checkpoint_config: CheckpointConfig,
    watchdog: LossSpikeWatchdog | None = None,
    on_metrics: Any | None = None,
    decay_phase: Any | None = None,
) -> dict[str, Any]:
    """Run the training loop with auto-rollback. Returns the final state.

    ``decay_phase`` is an optional ``DecayPhaseActivation`` from
    ``myllm.training.decay_phase``. When provided, the loop calls
    ``decay_phase.maybe_inject(state, batch)`` before each training step.
    In the stable phase this is a pass-through; in the decay phase the
    batch gets augmented with teacher top-K logits + indices, and the
    train_step's distillation-aware loss kicks in.
    """
    ckpt = CheckpointManager(checkpoint_config)

    # Ensure recovery multiplier is present.
    state = dict(initial_state)
    state.setdefault("lr_recovery_multiplier", 1.0)

    # Resume from latest complete checkpoint if any.
    resume_step = ckpt.latest_complete_step()
    if resume_step is not None:
        log.info("resuming_from_checkpoint", step=resume_step)
        restored = ckpt.restore(resume_step)
        state = {**state, **restored, "step": resume_step}
    else:
        log.info("starting_fresh", step=state.get("step", 0))

    target_steps = loop_config.total_steps
    recovery_count = 0
    consumed_iter: Iterator[dict[str, Any]] = iter(data_iter)
    _decay_phase_activated_logged = False

    for batch in consumed_iter:
        if state["step"] >= target_steps:
            break

        # Decay-phase distillation injection (no-op if disabled or pre-activation).
        if decay_phase is not None:
            if not _decay_phase_activated_logged and decay_phase.is_active(state):
                log.info(
                    "decay_phase_distillation_activated",
                    step=int(state["step"]),
                    activation_step=getattr(decay_phase, "activation_step", None),
                )
                _decay_phase_activated_logged = True
            batch = decay_phase.maybe_inject(state, batch)

        state, metrics = train_step_fn(state, batch)
        loss = float(metrics["loss"])

        # Watchdog ───────────────────────────────────────────────────────────
        if watchdog is not None:
            verdict = watchdog.observe(loss)
            if verdict == "hard":
                if recovery_count >= loop_config.max_recoveries:
                    log.error(
                        "spike_recovery_exhausted",
                        recoveries=recovery_count,
                        step=state["step"],
                        loss=loss,
                    )
                    raise LossSpikeError(
                        f"hard spike at step {state['step']} after "
                        f"{recovery_count} recoveries; loss={loss:.4f}"
                    )
                state, consumed_iter, recovery_count = _recover_from_spike(
                    state=state,
                    loss=loss,
                    ckpt=ckpt,
                    data_iter=consumed_iter,
                    watchdog=watchdog,
                    loop_config=loop_config,
                    recovery_count=recovery_count,
                )
                # Skip the rest of this iteration; the next batch starts after
                # the skipped chunk.
                continue
            if verdict == "soft":
                log.warning("soft_spike", step=state["step"], loss=loss)

        step_int = int(state["step"])

        # Logging ────────────────────────────────────────────────────────────
        if step_int % loop_config.log_every == 0:
            log.info(
                "step",
                step=step_int,
                loss=loss,
                lr_mult=float(state.get("lr_recovery_multiplier", 1.0)),
            )
            if on_metrics is not None:
                metrics_for_log = {
                    "loss": loss,
                    "lr_recovery_multiplier": float(
                        state.get("lr_recovery_multiplier", 1.0)
                    ),
                    **{k: float(v) for k, v in metrics.items() if k != "loss"},
                }
                on_metrics(step_int, metrics_for_log)

        # Checkpoint ─────────────────────────────────────────────────────────
        if step_int % loop_config.checkpoint_every == 0:
            ckpt.save(step_int, _state_to_save(state))

    # Final checkpoint, but skip if the cadence already saved it this step.
    final_step = int(state["step"])
    if final_step % loop_config.checkpoint_every != 0:
        ckpt.save(final_step, _state_to_save(state), extra={"reason": "final"})
    return state


def _recover_from_spike(
    *,
    state: dict[str, Any],
    loss: float,
    ckpt: CheckpointManager,
    data_iter: Iterator[dict[str, Any]],
    watchdog: LossSpikeWatchdog,
    loop_config: LoopConfig,
    recovery_count: int,
) -> tuple[dict[str, Any], Iterator[dict[str, Any]], int]:
    """Execute the rollback recovery. Returns (new_state, new_iter, new_count)."""
    spike_step = state["step"]
    log.warning(
        "hard_spike_initiating_recovery",
        step=spike_step,
        loss=loss,
        recovery_count=recovery_count + 1,
    )

    # 1. Save a failure marker so a post-mortem can find the bad point.
    try:
        ckpt.save(
            spike_step,
            _state_to_save(state),
            extra={"reason": "spike_marker", "loss": loss},
        )
    except Exception as e:  # noqa: BLE001
        log.error("spike_marker_save_failed", step=spike_step, error=str(e))

    # 2. Restore the most recent good checkpoint strictly before the spike.
    candidates = [s for s in ckpt.list_complete_steps() if s < spike_step]
    if not candidates:
        log.error("no_pre_spike_checkpoint_available", spike_step=spike_step)
        raise LossSpikeError(
            f"hard spike at step {spike_step}; no pre-spike checkpoint to restore"
        )
    rollback_to = max(candidates)
    restored = ckpt.restore(rollback_to)
    new_state = {**state, **restored, "step": rollback_to}

    # 3. Halve the recovery multiplier (compounds across recoveries).
    prev_mult = float(new_state.get("lr_recovery_multiplier", 1.0))
    new_mult = prev_mult * loop_config.recovery_lr_decay
    new_state["lr_recovery_multiplier"] = new_mult

    # 4. Skip a chunk of batches so the offending data range isn't replayed.
    skipped = 0
    for _ in range(loop_config.recovery_skip_batches):
        try:
            next(data_iter)
            skipped += 1
        except StopIteration:
            log.warning("data_iter_exhausted_during_skip", skipped=skipped)
            break

    # 5. Reset watchdog so post-restore noise doesn't immediately re-trigger.
    watchdog.reset()

    log.warning(
        "hard_spike_recovery_complete",
        rolled_back_to=rollback_to,
        new_lr_mult=new_mult,
        previous_lr_mult=prev_mult,
        skipped_batches=skipped,
    )

    return new_state, data_iter, recovery_count + 1


def _state_to_save(state: dict[str, Any]) -> dict[str, Any]:
    """Return the subset of state that gets persisted."""
    missing = [k for k in _PERSIST_KEYS if k not in state]
    if missing:
        raise TrainingError(
            f"state missing required keys for checkpoint: {missing}"
        )
    return {k: state[k] for k in _PERSIST_KEYS}
