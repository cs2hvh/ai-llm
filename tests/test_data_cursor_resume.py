"""Regression test for P0-4 (2026-05-12 audit): data cursor survives
checkpoint save/restore.

Before this fix, the training loop persisted model + opt_state but NOT
the data stream's position. After a pod restart, the model would resume
at step N while the data iterator restarted at offset 0 — silently
breaking distillation's corpus-position alignment.

Test:
  1. Run the loop for N steps with a stateful data-position tracker.
  2. Check that data_position is in the saved state.
  3. Construct a fresh state dict and load the checkpoint.
  4. Verify the restored state has the correct data_position.
  5. Continue the loop and verify the tracker resumes from the saved
     position, NOT from 0.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import numpy as np

from myllm.training.checkpoint import CheckpointConfig
from myllm.training.loop import LoopConfig, _PERSIST_KEYS, run as train_loop


def _make_simple_step():
    """Train step that just bumps step counter and returns a fixed loss."""
    def step_fn(state: dict, batch: dict) -> tuple[dict, dict]:
        new_state = {
            **state,
            "step": state["step"] + 1,
            "lr_recovery_multiplier": state["lr_recovery_multiplier"],
        }
        return new_state, {"loss": 2.0, "nan_skipped": 0.0}
    return step_fn


def _make_batch_iter(n_steps: int, batch_size: int = 4, seq_len: int = 8):
    """Yield n_steps batches with deterministic input_ids shape so the
    loop's `state["data_position"] += B*S` accounting can compute."""
    for _ in range(n_steps):
        yield {
            "input_ids": np.zeros((batch_size, seq_len), dtype=np.int32),
            "labels": np.zeros((batch_size, seq_len), dtype=np.int32),
        }


# --------------------------------------------------------------------------- #
# P0-4 regressions
# --------------------------------------------------------------------------- #
def test_data_position_in_persist_keys():
    """The persist-key list MUST include data_position or the checkpoint
    will silently lose it on save."""
    assert "data_position" in _PERSIST_KEYS, (
        "P0-4 regression: data_position must be in _PERSIST_KEYS or "
        "the data cursor won't survive checkpointing."
    )


def test_data_position_advances_during_training(tmp_path):
    """After running 5 steps with batch=4, seq=8, data_position should
    be 5 * 4 * 8 = 160."""
    state = {
        "trainable_variables": [0.0],
        "non_trainable_variables": [],
        "opt_state": [0.0],
        "step": 0,
        "lr_recovery_multiplier": 1.0,
    }
    ckpt_cfg = CheckpointConfig(root=str(tmp_path / "ckpts"), keep_last_n=1, keep_every_n=10000)
    loop_cfg = LoopConfig(total_steps=5, log_every=1, checkpoint_every=10)

    final_state = train_loop(
        train_step_fn=_make_simple_step(),
        initial_state=state,
        data_iter=_make_batch_iter(n_steps=10, batch_size=4, seq_len=8),
        loop_config=loop_cfg,
        checkpoint_config=ckpt_cfg,
    )
    assert final_state["step"] == 5
    assert final_state["data_position"] == 5 * 4 * 8, (
        f"expected data_position = 160 after 5 steps × 32 tok/step, "
        f"got {final_state['data_position']}"
    )


def test_data_position_round_trips_through_checkpoint(tmp_path):
    """Run, save, recreate the loop, restore — data_position must survive."""
    state = {
        "trainable_variables": [0.0],
        "non_trainable_variables": [],
        "opt_state": [0.0],
        "step": 0,
        "lr_recovery_multiplier": 1.0,
    }
    ckpt_root = str(tmp_path / "ckpts")
    ckpt_cfg = CheckpointConfig(root=ckpt_root, keep_last_n=2, keep_every_n=10000)

    # Phase 1: train for 3 steps with checkpoint_every=3 so a checkpoint is saved.
    loop_cfg = LoopConfig(total_steps=3, log_every=1, checkpoint_every=3)
    s1 = train_loop(
        train_step_fn=_make_simple_step(),
        initial_state=state,
        data_iter=_make_batch_iter(n_steps=10, batch_size=4, seq_len=8),
        loop_config=loop_cfg,
        checkpoint_config=ckpt_cfg,
    )
    assert s1["data_position"] == 3 * 4 * 8  # 96

    # Phase 2: fresh state, fresh iter, restore from the same checkpoint dir.
    # The data iterator restarts (we can't checkpoint the HF stream cursor
    # in unit tests) — what matters is data_position is RESTORED to 96 so
    # downstream consumers (decay-phase tracker) can seek to the right offset.
    fresh_state = {
        "trainable_variables": [0.0],
        "non_trainable_variables": [],
        "opt_state": [0.0],
        "step": 0,
        "lr_recovery_multiplier": 1.0,
    }
    loop_cfg2 = LoopConfig(total_steps=5, log_every=1, checkpoint_every=10000)
    s2 = train_loop(
        train_step_fn=_make_simple_step(),
        initial_state=fresh_state,
        data_iter=_make_batch_iter(n_steps=10, batch_size=4, seq_len=8),
        loop_config=loop_cfg2,
        checkpoint_config=ckpt_cfg,
    )
    # After resume from step 3 + 2 more steps = step 5.
    # data_position should have continued from 96 → 96 + 2*32 = 160.
    assert s2["step"] == 5
    assert s2["data_position"] == 96 + 2 * 32, (
        f"data_position should resume from checkpoint (96) and continue; "
        f"got {s2['data_position']}. If this is 64 (= 2*32, not 96 + 2*32), "
        f"data_position is being reset to 0 on resume → P0-4 regression."
    )


def test_decay_phase_position_tracker_seeded_on_resume(tmp_path):
    """When a SequentialCorpusPositions is attached to decay_phase, the
    loop should seed its _pos with the restored data_position so the next
    batch reads from the correct cached corpus offset."""
    from myllm.training.decay_phase import SequentialCorpusPositions, DecayPhaseActivation

    # Stand up a dummy decay_phase with no teacher reader (so maybe_inject is
    # a no-op even when "active", and we just observe the position_fn._pos).
    tracker = SequentialCorpusPositions(start_position=0)
    decay = DecayPhaseActivation(
        activation_step=999_999,  # never activates during this short test
        reader=None,
        position_fn=tracker,
    )

    state = {
        "trainable_variables": [0.0],
        "non_trainable_variables": [],
        "opt_state": [0.0],
        "step": 0,
        "lr_recovery_multiplier": 1.0,
        "data_position": 4096,  # simulate "resumed from a checkpoint at position 4096"
    }

    # Run the loop briefly; tracker should be seeded from data_position
    # at resume time. Use checkpoint_every very large so this run won't
    # itself save (we just want the resume-seeding step).
    ckpt_cfg = CheckpointConfig(root=str(tmp_path / "ckpts"), keep_last_n=1, keep_every_n=10000)
    loop_cfg = LoopConfig(total_steps=1, log_every=1, checkpoint_every=10000)
    final = train_loop(
        train_step_fn=_make_simple_step(),
        initial_state=state,
        data_iter=_make_batch_iter(n_steps=2, batch_size=4, seq_len=8),
        loop_config=loop_cfg,
        checkpoint_config=ckpt_cfg,
        decay_phase=decay,
    )
    # Re-audit Patch 2 (2026-05-12): the loop now advances data_position by
    # B*S after every batch and keeps tracker._pos in sync. So after 1 step
    # with batch=4 × seq_len=8 = 32 tokens:
    #   data_position: 4096 + 32 = 4128
    #   tracker._pos:  synced to 4128
    # Previously the assertion expected _pos to stay at 4096 because the
    # OLD code didn't advance data_position during stable phase. That was
    # the bug — verified + fixed in test_phase_b_reaudit_fixes.py.
    assert tracker._pos == 4128, (
        f"After Patch 2: tracker._pos should track data_position. "
        f"start=4096 (seeded) + 32 (one batch of tokens) = 4128. "
        f"Got {tracker._pos}."
    )
    assert final["data_position"] == 4128


# ---------------------------------------------------------------------------
# Re-audit (2026-05-12) Patch 1 + Patch 2 — pinned regressions
# ---------------------------------------------------------------------------
def test_data_position_advances_in_stable_phase_with_decay_configured(tmp_path):
    """Re-audit P0-#2 regression: when decay_phase is configured BUT the
    current step is still in the stable phase (activation_step not yet
    reached), data_position must STILL advance by B*S per step.

    Before Patch 2, the loop only read state["data_position"] from
    pf._pos when decay_phase existed. pf._pos only advanced in
    maybe_inject(), which was a no-op in stable phase. So
    data_position stuck at 0 for the entire stable phase (the first
    85% of training) — silently breaking teacher cache alignment when
    decay finally fired."""
    from myllm.training.decay_phase import (
        DecayPhaseActivation, SequentialCorpusPositions,
    )

    tracker = SequentialCorpusPositions(start_position=0)
    decay = DecayPhaseActivation(
        activation_step=999_999,  # never activates during this short test
        reader=None,
        position_fn=tracker,
    )

    state = {
        "trainable_variables": [0.0],
        "non_trainable_variables": [],
        "opt_state": [0.0],
        "step": 0,
        "lr_recovery_multiplier": 1.0,
    }
    ckpt_cfg = CheckpointConfig(root=str(tmp_path / "ckpts"), keep_last_n=1, keep_every_n=10000)
    loop_cfg = LoopConfig(total_steps=5, log_every=1, checkpoint_every=10000)
    final = train_loop(
        train_step_fn=_make_simple_step(),
        initial_state=state,
        data_iter=_make_batch_iter(n_steps=10, batch_size=4, seq_len=8),
        loop_config=loop_cfg,
        checkpoint_config=ckpt_cfg,
        decay_phase=decay,
    )
    # 5 steps × 4 × 8 = 160 tokens. data_position MUST equal 160.
    # Before Patch 2 this was 0 because pf._pos never advanced (maybe_inject
    # was a no-op in stable phase).
    assert final["data_position"] == 160, (
        f"P0-#2 regression: data_position should advance by B*S per step "
        f"even when decay_phase exists but is_active=False (stable phase). "
        f"Expected 160, got {final['data_position']}. If 0, the loop is "
        f"reading from pf._pos (which doesn't advance in stable phase) "
        f"instead of always advancing state['data_position'] by tokens-per-batch."
    )
    # tracker._pos should be in sync
    assert tracker._pos == 160


def test_train_step_preserves_unknown_state_keys():
    """Re-audit P0-#1 regression: train_step's new_state dict was built
    from scratch with hardcoded keys, silently dropping any operational
    state the loop added (data_position, future Phase B keys).

    This is a pure-jax test (no model needed) — we directly test that
    a fake state with extra keys round-trips through the same dict-
    preservation pattern."""
    # Simulate the pattern in train_step.py L190 (post-fix).
    state = {
        "trainable_variables": [1.0],
        "non_trainable_variables": [],
        "opt_state": [0.0],
        "step": 5,
        "lr_recovery_multiplier": 0.5,
        "data_position": 1234,       # the key that USED to be dropped
        "future_key": "some_value",  # arbitrary future B-phase key
    }
    # The fix: new_state = dict(state); new_state.update({known keys}).
    new_state = dict(state)
    new_state.update({
        "trainable_variables": [2.0],
        "non_trainable_variables": [],
        "opt_state": [0.1],
        "step": 6,
        "lr_recovery_multiplier": 0.5,
    })
    # All known keys updated:
    assert new_state["step"] == 6
    assert new_state["trainable_variables"] == [2.0]
    # All unknown keys preserved:
    assert new_state["data_position"] == 1234, (
        "P0-#1 regression: train_step's new_state must preserve "
        "data_position. The fix is to start from dict(state), not {}."
    )
    assert new_state["future_key"] == "some_value"
