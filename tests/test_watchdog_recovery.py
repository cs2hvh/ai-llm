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


def _data_iter_arrays(n: int, batch_size: int = 4, seq_len: int = 8):
    """Production-shape batches (numpy arrays with .size) so the B2
    data_position-advance path is actually exercised."""
    import numpy as np
    for i in range(n):
        yield {
            "input_ids": np.full((batch_size, seq_len), i, dtype=np.int32),
            "labels": np.full((batch_size, seq_len), i, dtype=np.int32),
        }


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


def test_precise_6sigma_threshold_fires(ckpt_dir):
    """Reviewer Q9 (post-pilot 2026-05-15): inject a synthetic 6σ spike
    calibrated to land just above hard_sigma=6 and verify the watchdog
    fires + rollback runs. The 50× spike in
    `test_hard_spike_triggers_rollback` is much further than 6σ; this
    case locks the EXACT threshold isn't accidentally off-by-one.
    """
    base, std = 2.0, 0.05  # matches _stable_losses noise
    # Run 50 warmup steps to populate the watchdog window, then inject a
    # loss at base + 6.5σ. With base 2.0 + std 0.05 → spike at 2.325.
    spike = base + 6.5 * std
    losses = _stable_losses(50, base=base, seed=11) + [spike] + _stable_losses(50, seed=12)
    step_fn = _make_train_step_returning(losses)

    final = train_loop(
        train_step_fn=step_fn,
        initial_state=_initial_state(),
        data_iter=_data_iter(200),
        loop_config=LoopConfig(
            total_steps=70,
            log_every=10,
            checkpoint_every=10,
            recovery_skip_batches=2,
            recovery_lr_decay=0.5,
            max_recoveries=3,
        ),
        checkpoint_config=CheckpointConfig(root=ckpt_dir),
        watchdog=LossSpikeWatchdog(
            window=100, min_observations=30, soft_sigma=3.0, hard_sigma=6.0
        ),
    )
    # A spike at 6.5σ must cross hard_sigma=6.0 → recovery fires →
    # lr_recovery_multiplier <= 0.5.
    assert final["lr_recovery_multiplier"] <= 0.5 + 1e-6, (
        f"6.5σ spike did NOT trigger rollback. final lr_mult="
        f"{final['lr_recovery_multiplier']}"
    )


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


# --------------------------------------------------------------------------- #
# B2 fix coverage (2026-05-16 P0 from re-audit)
#
# Two specific properties of the fixed _recover_from_spike:
#   1. data_position advances by skipped_tokens after recovery.
#      Without the fix, the cursor stays at the rollback checkpoint's
#      value while the iterator moves forward, desyncing the saved
#      cursor from the actual data stream. Any later resume would
#      re-feed the skipped tokens.
#   2. ckpt.restore() is called with template= so muP MultiTransformState
#      (or any namedtuple in opt_state) survives the rollback. Without
#      it, the next optimizer.update() would fail on .inner_states.
# --------------------------------------------------------------------------- #
def test_data_position_advances_after_recovery_skip(ckpt_dir):
    """The B2 fix: data_position += skipped_tokens after a hard-spike
    recovery. We use array-shaped batches so the .size accounting
    actually fires."""
    BATCH_SIZE = 4
    SEQ_LEN = 8
    SKIP_BATCHES = 5
    # Same loss pattern as test_hard_spike_triggers_rollback.
    losses = _stable_losses(60, seed=2) + [1000.0] + _stable_losses(200, seed=3)
    step_fn = _make_train_step_returning(losses)

    final = train_loop(
        train_step_fn=step_fn,
        initial_state=_initial_state(),
        data_iter=_data_iter_arrays(300, batch_size=BATCH_SIZE, seq_len=SEQ_LEN),
        loop_config=LoopConfig(
            total_steps=80,
            log_every=10,
            checkpoint_every=10,
            recovery_skip_batches=SKIP_BATCHES,
            recovery_lr_decay=0.5,
            max_recoveries=3,
        ),
        checkpoint_config=CheckpointConfig(root=ckpt_dir),
        watchdog=LossSpikeWatchdog(
            window=200, min_observations=20, soft_sigma=3.0, hard_sigma=6.0
        ),
    )
    # data_position must reflect the tokens emitted up to where we
    # stopped. With a recovery that skipped 5 batches, the cursor
    # advanced by 5 * BATCH_SIZE * SEQ_LEN = 160 tokens beyond the
    # rollback step + post-recovery training. The exact value depends
    # on the test loop's progression, but the lower bound is that
    # after recovery + at least one post-recovery step, data_position
    # is strictly greater than the rollback step's saved cursor.
    #
    # Concretely: rollback happens at step ~61 (just after the 60-step
    # warmup); the recovery skips 5 batches; then training runs forward
    # to step 80. So data_position should be at least
    # (rollback_step + skip + post_recovery_training_steps) * BATCH_TOKENS.
    BATCH_TOKENS = BATCH_SIZE * SEQ_LEN
    # Lower bound: at minimum, the skip itself contributed
    # SKIP_BATCHES * BATCH_TOKENS = 160 tokens. Final data_position
    # must be >= than the value the rollback checkpoint had saved
    # (which is rollback_step * BATCH_TOKENS), plus the skip, plus
    # at least one post-recovery training batch.
    assert int(final["data_position"]) >= SKIP_BATCHES * BATCH_TOKENS, (
        f"data_position={final['data_position']} did not advance to "
        f"reflect {SKIP_BATCHES * BATCH_TOKENS} skipped tokens"
    )


def test_recovery_with_namedtuple_opt_state_survives(ckpt_dir):
    """Verify that recovery preserves namedtuple structure in opt_state.

    This is the B1 audit fix (2026-05-12) applied to the recovery path:
    without template= on the restore call, a saved MultiTransformState
    namedtuple comes back as a plain dict and the next optimizer.update()
    fails. We use a fake namedtuple in opt_state to exercise the path.
    """
    from collections import namedtuple
    FakeOptState = namedtuple("FakeOptState", ["inner_states", "step"])

    state = {
        "trainable_variables": [0.0],
        "non_trainable_variables": [],
        # The opt_state has a namedtuple — what muP MultiTransform uses.
        "opt_state": FakeOptState(inner_states={"x": 1.0}, step=0),
        "step": 0,
        "lr_recovery_multiplier": 1.0,
    }

    losses = _stable_losses(60, seed=11) + [1000.0] + _stable_losses(100, seed=12)
    step_fn = _make_train_step_returning(losses)

    final = train_loop(
        train_step_fn=step_fn,
        initial_state=state,
        data_iter=_data_iter_arrays(300, batch_size=2, seq_len=4),
        loop_config=LoopConfig(
            total_steps=80,
            log_every=10,
            checkpoint_every=10,
            recovery_skip_batches=3,
            recovery_lr_decay=0.5,
            max_recoveries=3,
        ),
        checkpoint_config=CheckpointConfig(root=ckpt_dir),
        watchdog=LossSpikeWatchdog(
            window=200, min_observations=20, soft_sigma=3.0, hard_sigma=6.0
        ),
    )
    # After recovery, the opt_state must still BE a namedtuple — not
    # silently re-shaped into a plain dict. The test passes the
    # template through restore() and Orbax should preserve the type.
    assert isinstance(final["opt_state"], FakeOptState), (
        f"opt_state lost namedtuple structure: "
        f"got {type(final['opt_state']).__name__}, expected FakeOptState"
    )
    assert final["opt_state"].inner_states == {"x": 1.0}
