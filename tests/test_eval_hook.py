"""Tests for the eval-during-training hook (src/myllm/training/eval_hook.py
+ loop integration in src/myllm/training/loop.py).

No real model required — we mock train_step_fn and the model state. The
purpose is to verify:
  1. take_held_out_batches splits the iterator correctly + leaves the
     rest available for training.
  2. make_validation_loss_eval returns the right metrics dict shape +
     handles edge cases (empty held-out, all-NaN losses).
  3. The loop's eval_fn parameter actually gets called every
     loop_config.eval_every steps and its metrics get logged.
"""
from __future__ import annotations

import math
from typing import Any

import pytest

from myllm.training.eval_hook import (
    make_validation_loss_eval,
    take_held_out_batches,
)


# --------------------------------------------------------------------------- #
# take_held_out_batches
# --------------------------------------------------------------------------- #
class TestTakeHeldOutBatches:
    def test_splits_first_n_off(self):
        src = iter([{"i": i} for i in range(10)])
        held, rest = take_held_out_batches(src, 3)
        assert held == [{"i": 0}, {"i": 1}, {"i": 2}]
        assert [next(rest) for _ in range(3)] == [{"i": 3}, {"i": 4}, {"i": 5}]

    def test_handles_iterator_exhaustion(self):
        src = iter([{"i": 0}, {"i": 1}])
        held, rest = take_held_out_batches(src, 5)
        assert len(held) == 2
        with pytest.raises(StopIteration):
            next(rest)

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="n must be > 0"):
            take_held_out_batches(iter([]), 0)


# --------------------------------------------------------------------------- #
# make_validation_loss_eval
# --------------------------------------------------------------------------- #
class TestValidationLossEval:
    def _mock_step(self, losses_to_return):
        """Train-step mock that returns the next loss from a fixed list."""
        i = {"n": 0}

        def step(state, batch):
            loss = losses_to_return[i["n"] % len(losses_to_return)]
            i["n"] += 1
            return state, {"loss": loss}

        return step

    def test_returns_loss_and_perplexity(self):
        step = self._mock_step([2.0, 3.0, 4.0])
        batches = [{"x": j} for j in range(3)]
        eval_fn = make_validation_loss_eval(step, batches)
        out = eval_fn(step=1000, state={})
        assert set(out.keys()) == {"val_loss", "val_ppl", "val_n_batches"}
        assert abs(out["val_loss"] - 3.0) < 1e-9  # mean of 2, 3, 4
        assert abs(out["val_ppl"] - math.exp(3.0)) < 1e-6
        assert out["val_n_batches"] == 3.0

    def test_custom_label(self):
        eval_fn = make_validation_loss_eval(
            self._mock_step([2.0]), [{"x": 0}], label="held"
        )
        out = eval_fn(step=0, state={})
        assert "held_loss" in out and "held_ppl" in out
        assert "val_loss" not in out

    def test_skips_nan_losses_in_mean(self):
        step = self._mock_step([2.0, float("nan"), 4.0])
        batches = [{"x": j} for j in range(3)]
        eval_fn = make_validation_loss_eval(step, batches)
        out = eval_fn(step=0, state={})
        # mean of [2.0, 4.0] = 3.0 — NaN dropped
        assert abs(out["val_loss"] - 3.0) < 1e-9
        assert out["val_n_batches"] == 2.0

    def test_returns_none_if_all_nan(self):
        step = self._mock_step([float("nan"), float("nan")])
        batches = [{"x": j} for j in range(2)]
        eval_fn = make_validation_loss_eval(step, batches)
        out = eval_fn(step=0, state={})
        assert out is None

    def test_empty_held_out_raises_at_build_time(self):
        with pytest.raises(ValueError, match="at least one"):
            make_validation_loss_eval(self._mock_step([]), [])


# --------------------------------------------------------------------------- #
# Loop integration — verify eval_fn is called at the right cadence
# --------------------------------------------------------------------------- #
class TestLoopEvalIntegration:
    def _build_minimal_loop_inputs(self, n_steps: int = 10):
        """Build the smallest training loop call that exercises eval_fn."""
        from myllm.training.loop import LoopConfig
        from myllm.training.checkpoint import CheckpointConfig
        import tempfile

        # Mock train_step: takes state, batch; bumps step; returns finite loss.
        def train_step(state, batch):
            new_state = dict(state)
            new_state["step"] = state.get("step", 0) + 1
            return new_state, {"loss": 1.5}

        initial_state = {
            "step": 0,
            "trainable_variables": [],
            "non_trainable_variables": [],
            "opt_state": {},
        }

        # Iterator of dict-shaped batches; "input_ids" so loop can compute
        # data_position (B=1, S=16 → 16 tokens/batch).
        import numpy as np
        batches = [
            {"input_ids": np.zeros((1, 16), dtype=np.int32)} for _ in range(n_steps + 1)
        ]

        ckpt_root = tempfile.mkdtemp()
        loop_cfg = LoopConfig(
            total_steps=n_steps,
            log_every=100,           # don't spam logs in test
            checkpoint_every=999999, # don't write a checkpoint mid-test
            eval_every=3,            # eval every 3 steps
        )
        ckpt_cfg = CheckpointConfig(root=ckpt_root, keep_last_n=1)
        return train_step, initial_state, iter(batches), loop_cfg, ckpt_cfg

    def test_eval_fn_called_at_cadence(self, tmp_path, monkeypatch):
        from myllm.training import loop as loop_module
        train_step, init, batches, loop_cfg, ckpt_cfg = (
            self._build_minimal_loop_inputs(n_steps=10)
        )
        ckpt_cfg = type(ckpt_cfg)(root=str(tmp_path / "ckpt"), keep_last_n=1)

        calls: list[int] = []

        def eval_fn(step: int, state: dict[str, Any]) -> dict[str, float]:
            calls.append(step)
            return {"val_loss": 2.0, "val_ppl": math.exp(2.0)}

        loop_module.run(
            train_step_fn=train_step,
            initial_state=init,
            data_iter=batches,
            loop_config=loop_cfg,
            checkpoint_config=ckpt_cfg,
            eval_fn=eval_fn,
        )
        # eval_every=3, total_steps=10, skip step 0 → steps 3, 6, 9 (10 isn't called because the loop breaks on step >= target_steps)
        assert calls == [3, 6, 9]

    def test_eval_fn_none_doesnt_break(self, tmp_path):
        from myllm.training import loop as loop_module
        train_step, init, batches, loop_cfg, ckpt_cfg = (
            self._build_minimal_loop_inputs(n_steps=5)
        )
        ckpt_cfg = type(ckpt_cfg)(root=str(tmp_path / "ckpt"), keep_last_n=1)
        # No eval_fn passed — should run cleanly.
        loop_module.run(
            train_step_fn=train_step,
            initial_state=init,
            data_iter=batches,
            loop_config=loop_cfg,
            checkpoint_config=ckpt_cfg,
        )

    def test_eval_fn_exception_is_swallowed(self, tmp_path):
        from myllm.training import loop as loop_module
        train_step, init, batches, loop_cfg, ckpt_cfg = (
            self._build_minimal_loop_inputs(n_steps=8)
        )
        ckpt_cfg = type(ckpt_cfg)(root=str(tmp_path / "ckpt"), keep_last_n=1)

        def bad_eval(step, state):
            raise RuntimeError("eval blew up")

        # Training MUST complete despite eval errors.
        final = loop_module.run(
            train_step_fn=train_step,
            initial_state=init,
            data_iter=batches,
            loop_config=loop_cfg,
            checkpoint_config=ckpt_cfg,
            eval_fn=bad_eval,
        )
        assert final["step"] == 8

    def test_eval_metrics_get_forwarded_to_on_metrics(self, tmp_path):
        from myllm.training import loop as loop_module
        train_step, init, batches, loop_cfg, ckpt_cfg = (
            self._build_minimal_loop_inputs(n_steps=6)
        )
        ckpt_cfg = type(ckpt_cfg)(root=str(tmp_path / "ckpt"), keep_last_n=1)
        observed: list[tuple[int, dict[str, float]]] = []

        def on_metrics(step, metrics):
            observed.append((step, dict(metrics)))

        def eval_fn(step, state):
            return {"val_loss": 2.5, "val_ppl": math.exp(2.5), "val_n_batches": 4.0}

        loop_module.run(
            train_step_fn=train_step,
            initial_state=init,
            data_iter=batches,
            loop_config=loop_cfg,
            checkpoint_config=ckpt_cfg,
            eval_fn=eval_fn,
            on_metrics=on_metrics,
        )
        # We expect at least one "eval/*" entry forwarded.
        eval_emissions = [m for _, m in observed if any(k.startswith("eval/") for k in m)]
        assert eval_emissions, "no eval/* metrics forwarded to on_metrics"
        # Confirm the keys we expect
        last = eval_emissions[-1]
        assert "eval/val_loss" in last
        assert "eval/val_ppl" in last
        assert "eval/val_n_batches" in last
        assert abs(last["eval/val_loss"] - 2.5) < 1e-9

    def test_eval_fn_does_not_see_data_position_in_state(self, tmp_path):
        # 2026-05-16 hotfix regression: the loop pops `data_position` from
        # state before calling eval_fn (mirroring the train_step path).
        # This is necessary because the FSDP-safe `make_eval_step`'s JIT
        # in_shardings is built from `state_shardings`, which intentionally
        # excludes `data_position`. Without the pop, the state pytree
        # arriving at eval_step has 6 keys but the JIT contract expects 5
        # -> "different numbers of pytree children" ValueError on every
        # eval cycle under --fsdp.
        from myllm.training import loop as loop_module
        train_step, init, batches, loop_cfg, ckpt_cfg = (
            self._build_minimal_loop_inputs(n_steps=6)
        )
        ckpt_cfg = type(ckpt_cfg)(root=str(tmp_path / "ckpt"), keep_last_n=1)

        seen_keys_per_call: list[set[str]] = []

        def eval_fn(step: int, state: dict[str, Any]) -> dict[str, float]:
            # Capture the set of state keys the eval call sees.
            seen_keys_per_call.append(set(state.keys()))
            return {"val_loss": 1.0}

        final = loop_module.run(
            train_step_fn=train_step,
            initial_state=init,
            data_iter=batches,
            loop_config=loop_cfg,
            checkpoint_config=ckpt_cfg,
            eval_fn=eval_fn,
        )
        # eval was called at least once (eval_every=3, total_steps=6 -> step 3)
        assert seen_keys_per_call, "eval_fn never invoked"
        # The state arriving at eval_fn must NOT contain data_position —
        # the loop pops it before the call. This is what makes the FSDP
        # in_shardings contract (5 keys) compatible with the live state.
        for keys in seen_keys_per_call:
            assert "data_position" not in keys, (
                f"eval_fn saw data_position in state ({keys}); the loop's "
                f"pop-restore around eval is broken (regression of "
                f"2026-05-16 hotfix)."
            )
        # After training completes, the loop must have restored
        # data_position in the final state.
        assert "data_position" in final

