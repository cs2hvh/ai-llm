"""Tests for inspect_train_step_collectives (FSDP Commit F).

The helper lowers a JIT'd function, compiles it, and counts collective
ops in the HLO. Used by run_pretrain.py under MYLLM_DEBUG_HLO=1 to catch
the silent-FSDP-as-DDP bug class (XLA falling back to all-reduce on
grads when it should be reduce-scatter).

These tests just verify the helper's mechanics — it returns counts +
HLO text, doesn't crash, handles the no-collectives case. The actual
"is FSDP firing correctly" assertion runs in run_pretrain.py with the
real train_step + real mesh.
"""
from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import numpy as np
import pytest


def _mesh(data_size: int = 4):
    import jax
    from jax.sharding import Mesh
    devices = jax.devices()
    if len(devices) < data_size:
        pytest.skip(f"need {data_size} CPU devices via XLA_FLAGS")
    arr = np.asarray(devices[:data_size]).reshape(data_size, 1)
    return Mesh(arr, axis_names=("data", "model"))


class TestInspectCollectives:
    def test_returns_counts_dict_and_hlo_text(self):
        import jax
        import jax.numpy as jnp
        from myllm.training.mesh import inspect_train_step_collectives

        # Trivial JIT'd function: state * 2 (no collectives expected).
        @jax.jit
        def f(state, batch):
            return state * 2, jnp.zeros(())

        state = jnp.ones((4, 4), dtype=jnp.float32)
        batch = jnp.zeros((4,), dtype=jnp.float32)
        counts, hlo = inspect_train_step_collectives(f, state, batch)

        # Expected keys
        assert set(counts.keys()) == {
            "reduce_scatter", "all_reduce", "all_gather", "all_to_all"
        }
        # All counts non-negative ints
        for k, v in counts.items():
            assert isinstance(v, int) and v >= 0, f"{k}={v} not a non-neg int"
        # No collectives on a pointwise operation
        assert counts["reduce_scatter"] == 0
        assert counts["all_reduce"] == 0
        # HLO text exists
        assert isinstance(hlo, str) and len(hlo) > 0

    def test_sharded_input_can_trigger_collectives(self):
        # When inputs are sharded and the function reduces across the
        # sharded axis, XLA inserts collectives. We don't pin the EXACT
        # op type (CPU vs GPU lower the same logical sharded reduction
        # differently) — just that *some* collective ops appear.
        # The end-to-end smoke against run_pretrain.py with --fsdp
        # already exercises the all-reduce-rich case (98K+ on CPU).
        import jax
        import jax.numpy as jnp
        from jax.sharding import NamedSharding, PartitionSpec as P
        from myllm.training.mesh import inspect_train_step_collectives

        mesh = _mesh(4)
        sharded = NamedSharding(mesh, P("data"))
        replicated = NamedSharding(mesh, P())

        # Sum across a sharded axis -> XLA needs a collective.
        @jax.jit
        def f(x, _batch):
            return jnp.sum(x), jnp.zeros(())

        state = jax.device_put(
            jnp.arange(16, dtype=jnp.float32), sharded,
        )
        batch = jax.device_put(jnp.zeros((), dtype=jnp.float32), replicated)
        counts, hlo = inspect_train_step_collectives(f, state, batch)

        # At least one collective op (CPU may use all-reduce or all-gather;
        # don't pin the type).
        total_collectives = sum(counts.values())
        assert total_collectives >= 1, (
            f"sum-over-sharded triggered zero collectives; counts={counts}; "
            f"HLO head={hlo[:500]}"
        )

    def test_handles_pytree_state(self):
        # The real train_step takes a dict-of-pytrees state. Make sure
        # the helper handles that input shape (it just forwards to
        # jit_fn.lower, so this is a sanity check).
        import jax
        import jax.numpy as jnp
        from myllm.training.mesh import inspect_train_step_collectives

        @jax.jit
        def f(state, batch):
            new_state = {
                "trainable_variables": [x * 2.0 for x in state["trainable_variables"]],
                "step": state["step"] + 1,
            }
            return new_state, {"loss": jnp.zeros(())}

        state = {
            "trainable_variables": [
                jnp.ones((4,), dtype=jnp.float32),
                jnp.ones((8,), dtype=jnp.float32),
            ],
            "step": jnp.array(0, dtype=jnp.int32),
        }
        batch = jnp.zeros((4,), dtype=jnp.float32)
        counts, hlo = inspect_train_step_collectives(f, state, batch)
        assert isinstance(counts, dict)
        assert isinstance(hlo, str)
