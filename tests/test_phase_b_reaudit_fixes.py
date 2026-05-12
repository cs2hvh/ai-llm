"""Regression tests for the 2026-05-12 Phase B re-audit P0 fixes.

External reviewer found four real bugs in the post-Phase-B repo:

  1. train_step.py builds new_state from scratch, dropping unknown state
     keys like data_position. After every train_step the loop's
     data_position counter silently reset.
  2. Loop's data_position only advanced when decay_phase's
     SequentialCorpusPositions._pos advanced — and _pos only advanced
     inside maybe_inject(), which is a no-op in stable phase. Result:
     data_position stuck at 0 for first 85% of training when distillation
     was configured.
  3. shard_state() hardcoded only 4 state keys, dropping
     lr_recovery_multiplier and data_position.
  4. run_pretrain.py only read micro_batch_per_device from the data yaml,
     ignoring the model yaml. Proxy B's micro_batch=4 setting was being
     silently overridden by the data yaml's default.

This file pins regressions for all four.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
optax = pytest.importorskip("optax")

# Load run_pretrain as a module to test its helpers without invoking main().
_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))


def _get_resolver(name: str):
    """Pull a single helper out of run_pretrain.py without importing the
    full module (which has heavy keras deps)."""
    import ast
    src = (_SCRIPTS / "run_pretrain.py").read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            mod = ast.Module(body=[node], type_ignores=[])
            ns: dict = {"log": _make_silent_log()}
            exec(compile(mod, "<extracted>", "exec"), ns)
            return ns[name]
    raise RuntimeError(f"{name} not found")


def _make_silent_log():
    """Stub the structlog log object so extracted helpers don't crash on
    missing log.info / log.warning calls."""
    class _S:
        def __getattr__(self, _): return lambda *a, **k: None
    return _S()


# ===========================================================================
# Patch 1 — state preservation
# ===========================================================================
class TestShardStatePreservesAllKeys:
    """shard_state must preserve EVERY key in the input state, not just
    the four it used to hardcode. Re-audit P0-#3."""

    def test_lr_recovery_multiplier_survives(self):
        from myllm.training.mesh import shard_state
        state = {
            "trainable_variables": [jnp.array([1.0])],
            "non_trainable_variables": [],
            "opt_state": [jnp.array([0.0])],
            "step": 0,
            "lr_recovery_multiplier": jnp.float32(0.5),
        }
        sharded = shard_state(state, replicate_sharding=None)
        assert "lr_recovery_multiplier" in sharded, (
            "shard_state dropped lr_recovery_multiplier. Re-audit "
            "regression: this used to be hardcoded out of the return dict."
        )
        assert float(sharded["lr_recovery_multiplier"]) == pytest.approx(0.5)

    def test_data_position_survives(self):
        from myllm.training.mesh import shard_state
        state = {
            "trainable_variables": [jnp.array([1.0])],
            "non_trainable_variables": [],
            "opt_state": [jnp.array([0.0])],
            "step": 0,
            "lr_recovery_multiplier": jnp.float32(1.0),
            "data_position": 1234,
        }
        sharded = shard_state(state, replicate_sharding=None)
        assert sharded.get("data_position") == 1234

    def test_arbitrary_extra_keys_survive(self):
        """Future Phase B work will add more state keys (per-source token
        counts, validation timer, etc.). shard_state must be generic."""
        from myllm.training.mesh import shard_state
        state = {
            "trainable_variables": [jnp.array([1.0])],
            "non_trainable_variables": [],
            "opt_state": [jnp.array([0.0])],
            "step": 0,
            "lr_recovery_multiplier": jnp.float32(1.0),
            "data_position": 0,
            "future_phase_b_key": "some metadata string",
            "another_one": 42,
        }
        sharded = shard_state(state, replicate_sharding=None)
        assert sharded["future_phase_b_key"] == "some metadata string"
        assert sharded["another_one"] == 42


# ===========================================================================
# Patch 3 — micro_batch resolver priority
# ===========================================================================
class TestResolveMicroBatch:
    """Re-audit P0-#4: micro_batch was only read from data yaml. Now must
    follow priority CLI > model yaml > data yaml > default."""

    def setup_method(self):
        self.resolve = _get_resolver("resolve_micro_batch")

    def test_cli_override_wins(self):
        result = self.resolve(
            cli_override=2,
            model_yaml={"batch": {"micro_batch_per_device": 4}},
            data_yaml={"batch": {"micro_batch_per_device": 8}},
        )
        assert result == 2

    def test_model_yaml_overrides_data_yaml(self):
        """The fix to the actual bug: Proxy B's model yaml says 4, data
        yaml says 8. Model yaml must win."""
        result = self.resolve(
            cli_override=None,
            model_yaml={"batch": {"micro_batch_per_device": 4}},
            data_yaml={"batch": {"micro_batch_per_device": 8}},
        )
        assert result == 4, (
            "P0-#4 regression: model yaml's micro_batch_per_device must "
            "override data yaml's. Proxy B sets 4; data yaml sets 8."
        )

    def test_data_yaml_used_when_no_model_setting(self):
        result = self.resolve(
            cli_override=None,
            model_yaml={"context_length": 4096},  # no batch block
            data_yaml={"batch": {"micro_batch_per_device": 8}},
        )
        assert result == 8

    def test_default_when_nothing_set(self):
        result = self.resolve(
            cli_override=None,
            model_yaml={"context_length": 4096},
            data_yaml={},
            default=16,
        )
        assert result == 16


# ===========================================================================
# Patch 3b — wind_tunnel_sweep must use the same resolver
# ===========================================================================
class TestSweepUsesModelMicroBatch:
    """Re-audit P0-#4 second half: wind_tunnel_sweep.py used to hardcode
    micro_batch=8 in cell_command, ignoring the model yaml. Now it reads
    from the same precedence: model > data > default."""

    def test_sweep_reads_proxy_b_micro_batch(self, tmp_path):
        # Import the sweep module via importlib (it's a script, not on path).
        # Must register in sys.modules BEFORE exec so dataclass introspection
        # of SweepCell can find the module.
        _spec = importlib.util.spec_from_file_location(
            "wind_tunnel_sweep_test_b", _SCRIPTS / "wind_tunnel_sweep.py"
        )
        wt = importlib.util.module_from_spec(_spec)
        sys.modules["wind_tunnel_sweep_test_b"] = wt
        _spec.loader.exec_module(wt)

        # Write a fake Proxy B model yaml with micro_batch=4.
        model_yaml = tmp_path / "proxy_b.yaml"
        model_yaml.write_text(
            "context_length: 4096\n"
            "batch:\n"
            "  micro_batch_per_device: 4\n"
        )
        # Write a fake data yaml with the typical 8.
        data_yaml = tmp_path / "data.yaml"
        data_yaml.write_text(
            "batch:\n"
            "  micro_batch_per_device: 8\n"
        )

        cell = wt.SweepCell("c1", 1e-3, 0.02, tokens=10_000_000)
        cmd = wt.cell_command(
            cell, str(model_yaml), str(data_yaml),
            "tok.json", "ckpts", "log.txt",
        )

        # The launched command must pass --micro-batch-override 4 to
        # run_pretrain.py, not 8.
        idx = cmd.index("--micro-batch-override")
        assert cmd[idx + 1] == "4", (
            "wind_tunnel_sweep.cell_command should pass model yaml's "
            f"micro_batch_per_device=4 down to run_pretrain, got {cmd[idx + 1]}"
        )

        # And total_steps math should also reflect micro_batch=4 (not 8):
        # 10M / (4 × 4096) = 610.35 → 611
        steps_idx = cmd.index("--total-steps")
        assert int(cmd[steps_idx + 1]) == pytest.approx(611, abs=1), (
            f"total_steps math must use the resolved micro_batch=4, not the "
            f"old hardcoded 8. Got {cmd[steps_idx + 1]}"
        )
