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
    # After 1 step with batch=4 × seq_len=8 = 32 tokens consumed.
    # tracker._pos at start = 4096 (seeded), then advances by 32 per call
    # because the loop reads it from the tracker.
    # The SequentialCorpusPositions only updates _pos when its __call__ runs,
    # which happens via maybe_inject — but decay is_active is False here
    # (activation_step=999_999). So _pos should stay at 4096 from seeding.
    assert tracker._pos == 4096, (
        f"tracker._pos should be seeded with data_position from state "
        f"on resume; got {tracker._pos}. If 0, the seeding path in "
        f"loop.run() didn't fire."
    )
    # data_position in the final state should equal tracker._pos when the
    # tracker is attached (loop reads from tracker when available).
    assert final["data_position"] == 4096
