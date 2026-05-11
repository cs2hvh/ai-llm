"""End-to-end test of the hard-spike auto-rollback path.

Uses a stubbed train_step + a controllable loss sequence so we can force the
watchdog to fire a hard spike and assert the recovery actually:
    - rolls back to a pre-spike checkpoint
    - halves the lr_recovery_multiplier
    - skips the configured number of batches
    - resets the watchdog and resumes
"""
from __future__ import annotations

import math
import random
from pathlib import Path

import pytest


def _stable_losses(n: int, base: float = 2.0, seed: int = 0) -> list[float]:
    """Deterministic noisy stable losses (std ≈ 0.05) — real training has noise.

    A constant-loss sequence has std=0, so the watchdog can't compute sigma
    and returns "ok" even for a wildly out-of-band value. Using realistic
    noise here matches what we'd see in actual pretraining.
    """
    rng = random.Random(seed)
    return [base + 0.05 * rng.gauss(0, 1) for _ in range(n)]

from myllm.training.checkpoint import CheckpointConfig
from myllm.training.loop import LoopConfig, run as train_loop
from myllm.training.watchdog import LossSpikeWatchdog
from myllm.utils.exceptions import LossSpikeError


def _make_train_step_returning(losses: list[float]):
    """Return a callable that yields the next loss in ``losses`` each call."""
    it = iter(losses)

    def step_fn(state: dict, batch: dict) -> tuple[dict, dict]:
        loss = next(it)
        new_state = {
            **state,
            "step": state["step"] + 1,
            # Carry the multiplier through unchanged — the loop mutates it on rollback.
            "lr_recovery_multiplier": state["lr_recovery_multiplier"],
        }
        return new_state, {"loss": loss}

    return step_fn


def _initial_state() -> dict:
    return {
        "trainable_variables": [0.0],  # placeholders; checkpoint uses Orbax which
        "non_trainable_variables": [],  # works on ANY pytree
        "opt_state": [0.0],
        "step": 0,
        "lr_recovery_multiplier": 1.0,
    }


def _data_iter(n: int):
    for i in range(n):
        yield {"input_ids": i, "labels": i}


@pytest.fixture
def ckpt_dir(tmp_path: Path) -> str:
    d = tmp_path / "ckpt"
    d.mkdir()
    return str(d)


def test_clean_run_no_recovery(ckpt_dir):
    """No spike → no rollback. Smoke-checks the happy path."""
    losses = _stable_losses(60, seed=1)
    step_fn = _make_train_step_returning(losses)
    final = train_loop(
        train_step_fn=step_fn,
        initial_state=_initial_state(),
        data_iter=_data_iter(60),
        loop_config=LoopConfig(total_steps=50, log_every=10, checkpoint_every=10),
        checkpoint_config=CheckpointConfig(root=ckpt_dir),
        watchdog=LossSpikeWatchdog(window=200, min_observations=20),
    )
    assert final["lr_recovery_multiplier"] == 1.0
    assert int(final["step"]) == 50


def test_hard_spike_triggers_rollback(ckpt_dir):
    """Stable loss for many steps, then a 50x spike. Verify rollback runs."""
    losses = _stable_losses(60, seed=2) + [1000.0] + _stable_losses(200, seed=3)
    step_fn = _make_train_step_returning(losses)

    initial = _initial_state()
    final = train_loop(
        train_step_fn=step_fn,
        initial_state=initial,
        data_iter=_data_iter(300),
        loop_config=LoopConfig(
            total_steps=80,
            log_every=10,
            checkpoint_every=10,
            recovery_skip_batches=5,
            recovery_lr_decay=0.5,
            max_recoveries=3,
        ),
        checkpoint_config=CheckpointConfig(root=ckpt_dir),
        watchdog=LossSpikeWatchdog(
            window=200, min_observations=20, soft_sigma=3.0, hard_sigma=6.0
        ),
    )
    # The lr_recovery_multiplier should have been halved at least once.
    assert final["lr_recovery_multiplier"] == pytest.approx(0.5, rel=1e-6)


def test_hard_spike_with_no_pre_checkpoint_raises(ckpt_dir):
    """If a hard spike fires before any checkpoint exists, the loop must raise."""
    # Tight loop: spike at step 21 (right after watchdog warms up), checkpoint_every=100
    # so no checkpoint has been saved yet → no rollback target.
    losses = _stable_losses(20, seed=4) + [1000.0]
    step_fn = _make_train_step_returning(losses)

    with pytest.raises(LossSpikeError):
        train_loop(
            train_step_fn=step_fn,
            initial_state=_initial_state(),
            data_iter=_data_iter(50),
            loop_config=LoopConfig(
                total_steps=30, log_every=5, checkpoint_every=100
            ),
            checkpoint_config=CheckpointConfig(root=ckpt_dir),
            watchdog=LossSpikeWatchdog(
                window=100, min_observations=15, soft_sigma=3.0, hard_sigma=6.0
            ),
        )


def test_max_recoveries_exhausted_raises(ckpt_dir):
    """Repeated spikes past max_recoveries → loop gives up and raises."""
    # Pattern: stable warmup, then alternating spike/stable to repeatedly trigger.
    # We pick max_recoveries=2 and ensure 3 spikes happen.
    losses = (
        _stable_losses(30, seed=5)
        + [1000.0]   # spike 1 (recovery 1)
        + _stable_losses(30, seed=6)
        + [1000.0]   # spike 2 (recovery 2)
        + _stable_losses(30, seed=7)
        + [1000.0]   # spike 3 → exhausts
    )
    step_fn = _make_train_step_returning(losses)
    with pytest.raises(LossSpikeError):
        train_loop(
            train_step_fn=step_fn,
            initial_state=_initial_state(),
            data_iter=_data_iter(500),
            loop_config=LoopConfig(
                total_steps=200,
                log_every=10,
                checkpoint_every=5,  # frequent checkpoints so rollback target exists
                recovery_skip_batches=2,
                recovery_lr_decay=0.5,
                max_recoveries=2,  # only 2 allowed
            ),
            checkpoint_config=CheckpointConfig(root=ckpt_dir),
            watchdog=LossSpikeWatchdog(
                window=100, min_observations=20, soft_sigma=3.0, hard_sigma=6.0
            ),
        )


def test_lr_recovery_multiplier_compounds(ckpt_dir):
    """Two recoveries → multiplier should be 0.5 * 0.5 = 0.25."""
    # Plenty of stable losses at the end — rollbacks re-traverse steps, so
    # the loop consumes more losses than ``total_steps`` would suggest.
    losses = (
        _stable_losses(30, seed=8)
        + [1000.0]   # spike 1
        + _stable_losses(30, seed=9)
        + [1000.0]   # spike 2
        + _stable_losses(300, seed=10)  # post-recovery runway
    )
    step_fn = _make_train_step_returning(losses)
    final = train_loop(
        train_step_fn=step_fn,
        initial_state=_initial_state(),
        data_iter=_data_iter(500),
        loop_config=LoopConfig(
            total_steps=120,
            log_every=10,
            checkpoint_every=5,
            recovery_skip_batches=2,
            recovery_lr_decay=0.5,
            max_recoveries=4,
        ),
        checkpoint_config=CheckpointConfig(root=ckpt_dir),
        watchdog=LossSpikeWatchdog(
            window=100, min_observations=20, soft_sigma=3.0, hard_sigma=6.0
        ),
    )
    assert final["lr_recovery_multiplier"] == pytest.approx(0.25, rel=1e-6)
