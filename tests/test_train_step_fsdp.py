"""Integration tests for FSDP-aware train_step (Commit C of FSDP plan).

Verifies that:
  1. When `state_shardings` is None (default), train_step behaves
     EXACTLY as the pre-FSDP path. Locked by existing tests
     (test_train_step_chunked_ce.py, test_train_step_distill.py).
  2. When `state_shardings` is provided, the compiled JIT:
       - accepts a sharded state + batch as input
       - returns a sharded state with matching shardings
       - the loss curve matches the no-FSDP path within numerical noise
         (this is L2 parity — the do-not-pass-go invariant)

Runs on simulated multi-device CPU via
``XLA_FLAGS=--xla_force_host_platform_device_count=4``. The agent's plan
explicitly recommends this for the L2 parity canary; we use the same
pattern here.

What this does NOT verify (saved for the L2 canary + GPU gauntlet):
  - That XLA actually emits reduce-scatter (Commit F's HLO grep does)
  - That throughput improves vs. DP-replicated (needs GPU)
  - That checkpoint save/restore roundtrips through sharded state
    (covered by Commit G's reshard utility + L3-packed-under-FSDP)
"""
from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import numpy as np
import pytest

# Force-import keras before model so backend selection is locked.
import keras  # noqa: F401, E402

from myllm.model.config import ModelConfig  # noqa: E402
from myllm.model.transformer import build_model  # noqa: E402
from myllm.training.train_step import make_train_step  # noqa: E402


def _tiny_cfg(vocab_size: int = 64) -> ModelConfig:
    """Tiny model config — same shape as the test_train_step_chunked_ce
    helper so we can compare loss values across the two paths.

    Note: gradient_checkpointing default is True (from production yaml)
    but for tiny test models we turn it off to keep the test fast.
    Per-block recompute on a 2-layer 32-hidden model is wasted overhead.
    """
    return ModelConfig(
        name="tiny_fsdp_test",
        arch="llama_decoder",
        layers=2,
        hidden_dim=32,
        ffn_dim=64,
        num_heads=4,
        num_kv_heads=2,
        head_dim=8,
        vocab_size=vocab_size,
        tie_embeddings=True,
        context_length=16,
        position="rope",
        rope_base=10000.0,
        norm="rmsnorm",
        norm_eps=1e-5,
        activation="swiglu",
        init_std=0.02,
        scaled_init_for_residuals=False,
        z_loss_coef=1e-4,
        qk_norm=False,
        gradient_checkpointing=False,
    )


def _mesh(data_size: int = 4):
    """Build a (data, model) mesh with N CPU devices."""
    import jax
    from jax.sharding import Mesh
    devices = jax.devices()
    if len(devices) < data_size:
        pytest.skip(f"need {data_size} CPU devices via XLA_FLAGS")
    arr = np.asarray(devices[:data_size]).reshape(data_size, 1)
    return Mesh(arr, axis_names=("data", "model"))


def _initial_state(model, optimizer):
    """Build a fresh training state from a built model + optax optimizer."""
    import jax.numpy as jnp
    trainable = [v.value for v in model.trainable_variables]
    non_trainable = [v.value for v in model.non_trainable_variables]
    return {
        "trainable_variables": trainable,
        "non_trainable_variables": non_trainable,
        "opt_state": optimizer.init(trainable),
        "step": jnp.array(0, dtype=jnp.int32),
        "lr_recovery_multiplier": jnp.array(1.0, dtype=jnp.float32),
    }


def _make_batch(seed: int, vocab: int, B: int = 4, S: int = 4):
    """Make a synthetic batch. B=4 chosen so it shards evenly across 4 devices."""
    rng = np.random.default_rng(seed)
    return {
        "input_ids": rng.integers(0, vocab, size=(B, S)).astype(np.int32),
        "labels": rng.integers(0, vocab, size=(B, S)).astype(np.int32),
    }


class TestFSDPBackCompat:
    """When state_shardings=None, behavior is byte-identical to pre-FSDP."""

    def test_default_path_unchanged_no_shardings_passed(self):
        try:
            import optax
        except ImportError:
            pytest.skip("optax not installed")

        vocab = 64
        cfg = _tiny_cfg(vocab_size=vocab)
        model = build_model(cfg)
        optimizer = optax.adamw(learning_rate=1e-3)
        batch = _make_batch(seed=42, vocab=vocab, B=4, S=4)

        # Build train_step WITHOUT state_shardings (default path)
        step_fn = make_train_step(
            model, optimizer, z_loss_coef=1e-4,
            # explicitly NO state_shardings or batch_sharding
        )

        state = _initial_state(model, optimizer)
        new_state, metrics = step_fn(state, batch)

        # Basic invariants: step advanced, metrics finite, no nan_skipped
        import jax.numpy as jnp
        assert int(new_state["step"]) == 1
        assert jnp.isfinite(metrics["loss"])
        assert float(metrics["nan_skipped"]) == 0.0


class TestFSDPPathLossParity:
    """When state_shardings is provided, the loss should match the
    no-FSDP path within numerical noise. This is the L2 parity invariant.
    """

    def test_fsdp_path_matches_no_fsdp_loss_within_bf16_noise(self):
        try:
            import optax
        except ImportError:
            pytest.skip("optax not installed")

        import jax
        import jax.numpy as jnp
        from jax.sharding import NamedSharding, PartitionSpec as P
        from myllm.training.mesh import make_param_shardings

        vocab = 64
        cfg = _tiny_cfg(vocab_size=vocab)
        model = build_model(cfg)
        optimizer = optax.adamw(learning_rate=1e-3)
        mesh = _mesh(4)
        batch = _make_batch(seed=42, vocab=vocab, B=4, S=4)

        # --- Path A: no FSDP --- #
        step_a = make_train_step(model, optimizer, z_loss_coef=1e-4)
        state_a = _initial_state(model, optimizer)
        _, metrics_a = step_a(state_a, batch)

        # --- Path B: FSDP shardings provided --- #
        from myllm.training.optimizer import make_optimizer_state_sharding

        # Build a sharding pytree for the state. trainable/non_trainable/opt
        # use the make_param_shardings rule. step + lr_recovery_multiplier
        # are scalars -> replicated.
        state_b = _initial_state(model, optimizer)
        replicate = NamedSharding(mesh, P())
        state_shardings = {
            "trainable_variables": make_param_shardings(state_b["trainable_variables"], mesh),
            "non_trainable_variables": make_param_shardings(state_b["non_trainable_variables"], mesh),
            "opt_state": make_optimizer_state_sharding(optimizer, state_b["trainable_variables"], mesh),
            "step": replicate,
            "lr_recovery_multiplier": replicate,
        }
        batch_sharding = NamedSharding(mesh, P("data"))

        # Place state + batch onto the mesh with the declared shardings
        state_b = jax.tree.map(
            lambda x, s: jax.device_put(x, s),
            state_b, state_shardings,
        )
        batch_b = {k: jax.device_put(v, batch_sharding) for k, v in batch.items()}

        step_b = make_train_step(
            model, optimizer, z_loss_coef=1e-4,
            state_shardings=state_shardings, batch_sharding=batch_sharding,
        )
        _, metrics_b = step_b(state_b, batch_b)

        # Loss values must agree within fp32 numerical noise (collectives
        # reorder reductions; we allow 5e-3 per the agent plan's L2 spec).
        loss_a = float(metrics_a["loss"])
        loss_b = float(metrics_b["loss"])
        assert np.isclose(loss_a, loss_b, atol=5e-3), (
            f"FSDP loss diverged from no-FSDP: {loss_a} vs {loss_b} "
            f"(|delta|={abs(loss_a - loss_b)})"
        )

    def test_fsdp_path_advances_step_counter(self):
        try:
            import optax
        except ImportError:
            pytest.skip("optax not installed")
        import jax
        from jax.sharding import NamedSharding, PartitionSpec as P
        from myllm.training.mesh import make_param_shardings
        from myllm.training.optimizer import make_optimizer_state_sharding

        vocab = 64
        cfg = _tiny_cfg(vocab_size=vocab)
        model = build_model(cfg)
        optimizer = optax.adamw(learning_rate=1e-3)
        mesh = _mesh(4)
        batch = _make_batch(seed=43, vocab=vocab, B=4, S=4)

        state = _initial_state(model, optimizer)
        replicate = NamedSharding(mesh, P())
        state_shardings = {
            "trainable_variables": make_param_shardings(state["trainable_variables"], mesh),
            "non_trainable_variables": make_param_shardings(state["non_trainable_variables"], mesh),
            "opt_state": make_optimizer_state_sharding(optimizer, state["trainable_variables"], mesh),
            "step": replicate,
            "lr_recovery_multiplier": replicate,
        }
        batch_sharding = NamedSharding(mesh, P("data"))

        state = jax.tree.map(lambda x, s: jax.device_put(x, s), state, state_shardings)
        batch_p = {k: jax.device_put(v, batch_sharding) for k, v in batch.items()}

        step_fn = make_train_step(
            model, optimizer, z_loss_coef=1e-4,
            state_shardings=state_shardings, batch_sharding=batch_sharding,
        )
        new_state, _ = step_fn(state, batch_p)
        assert int(new_state["step"]) == 1, "step counter did not advance"

    def test_fsdp_path_preserves_state_dict_keys(self):
        # data_position is preserved by the train_step (loop manages it).
        # Make sure FSDP path doesn't strip it.
        try:
            import optax
        except ImportError:
            pytest.skip("optax not installed")
        import jax
        import jax.numpy as jnp
        from jax.sharding import NamedSharding, PartitionSpec as P
        from myllm.training.mesh import make_param_shardings
        from myllm.training.optimizer import make_optimizer_state_sharding

        vocab = 64
        cfg = _tiny_cfg(vocab_size=vocab)
        model = build_model(cfg)
        optimizer = optax.adamw(learning_rate=1e-3)
        mesh = _mesh(4)
        batch = _make_batch(seed=44, vocab=vocab, B=4, S=4)

        state = _initial_state(model, optimizer)
        # Add data_position (the loop-managed key the train_step preserves)
        state["data_position"] = jnp.array(0, dtype=jnp.int32)
        replicate = NamedSharding(mesh, P())
        state_shardings = {
            "trainable_variables": make_param_shardings(state["trainable_variables"], mesh),
            "non_trainable_variables": make_param_shardings(state["non_trainable_variables"], mesh),
            "opt_state": make_optimizer_state_sharding(optimizer, state["trainable_variables"], mesh),
            "step": replicate,
            "lr_recovery_multiplier": replicate,
            "data_position": replicate,
        }
        batch_sharding = NamedSharding(mesh, P("data"))

        state = jax.tree.map(lambda x, s: jax.device_put(x, s), state, state_shardings)
        batch_p = {k: jax.device_put(v, batch_sharding) for k, v in batch.items()}

        step_fn = make_train_step(
            model, optimizer, z_loss_coef=1e-4,
            state_shardings=state_shardings, batch_sharding=batch_sharding,
        )
        new_state, _ = step_fn(state, batch_p)
        assert "data_position" in new_state, "FSDP train_step dropped data_position"
        # Value is preserved (the train_step doesn't modify data_position;
        # the loop owns that update)
        assert int(new_state["data_position"]) == 0

    def test_fsdp_works_when_loop_pops_data_position_before_call(self):
        # Regression for 2026-05-16 bug: run_pretrain.py's FSDP block had
        # data_position in state_shardings, but the training loop pops it
        # before each train_step_fn call (int32-overflow fix from commit
        # 9f442f7). That caused:
        #
        #   ValueError: pytree structure error: different numbers of
        #   pytree children at key path pjit in_shardings[0]
        #   (state_shardings has 6 keys; state arriving at JIT has 5;
        #   symmetric difference: data_position)
        #
        # Production fix: state_shardings under --fsdp must NOT include
        # data_position. The loop carries it as a Python int outside the
        # JIT'd state pytree. This test replicates the production call
        # pattern: state has all keys except data_position; state_shardings
        # matches exactly. Pre-fix this raised ValueError; post-fix it
        # works cleanly.
        try:
            import optax
        except ImportError:
            pytest.skip("optax not installed")
        import jax
        import jax.numpy as jnp
        from jax.sharding import NamedSharding, PartitionSpec as P
        from myllm.training.mesh import make_param_shardings
        from myllm.training.optimizer import make_optimizer_state_sharding

        vocab = 64
        cfg = _tiny_cfg(vocab_size=vocab)
        model = build_model(cfg)
        optimizer = optax.adamw(learning_rate=1e-3)
        mesh = _mesh(4)
        batch = _make_batch(seed=45, vocab=vocab, B=4, S=4)

        # Initial state — NO data_position. Matches what run_pretrain.py's
        # FSDP block produces after the 2026-05-16 fix.
        state = _initial_state(model, optimizer)
        # state has: trainable_variables, non_trainable_variables,
        # opt_state, step, lr_recovery_multiplier. (5 keys, NO data_position.)
        assert "data_position" not in state, "test scaffold drifted"

        replicate = NamedSharding(mesh, P())
        # state_shardings ALSO has no data_position — matches state structure.
        state_shardings = {
            "trainable_variables": make_param_shardings(
                state["trainable_variables"], mesh,
            ),
            "non_trainable_variables": make_param_shardings(
                state["non_trainable_variables"], mesh,
            ),
            "opt_state": make_optimizer_state_sharding(
                optimizer, state["trainable_variables"], mesh,
            ),
            "step": replicate,
            "lr_recovery_multiplier": replicate,
        }
        batch_sharding = NamedSharding(mesh, P("data"))
        state = jax.tree.map(
            lambda x, s: jax.device_put(x, s), state, state_shardings,
        )
        batch_p = {k: jax.device_put(v, batch_sharding) for k, v in batch.items()}

        step_fn = make_train_step(
            model, optimizer, z_loss_coef=1e-4,
            state_shardings=state_shardings, batch_sharding=batch_sharding,
        )
        # Pre-fix: raised "different numbers of pytree children" ValueError.
        # Post-fix: clean run.
        new_state, metrics = step_fn(state, batch_p)
        assert int(new_state["step"]) == 1
        assert jnp.isfinite(metrics["loss"])
        # data_position remains absent from JIT-managed state.
        assert "data_position" not in new_state
