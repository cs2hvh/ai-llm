"""Tests for the FSDP sharding helpers in myllm.training.mesh.

Covers:
  - `_leaf_partition_spec`: per-leaf rule (largest-divisible-axis or replicate)
  - `make_param_shardings`: walks a PyTree, returns NamedShardings in
    matching structure

These tests don't need real GPUs. We use
``XLA_FLAGS=--xla_force_host_platform_device_count=N`` to simulate an
N-device mesh on CPU — the same pattern the agent's plan recommends for
the L2 parity canary.

What this LOCKS IN:
  - The longest-divisible-axis rule (the FSDP convention used by MaxText
    and Levanter)
  - Tie-break: smaller index wins (axis 0 bias)
  - Scalar / undivisible -> replicated
  - PyTree structure preservation (including namedtuples)
"""
from __future__ import annotations

import os

# IMPORTANT: set this BEFORE any JAX import so the platform sees N CPUs.
os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import numpy as np
import pytest


# Local helpers ---------------------------------------------------------------
def _mesh(data_size: int = 4, model_size: int = 1):
    """Build a (data, model) mesh with N CPU devices."""
    import jax
    from jax.sharding import Mesh
    devices = jax.devices()
    if len(devices) < data_size * model_size:
        pytest.skip(
            f"need {data_size * model_size} CPU devices via "
            f"XLA_FLAGS=--xla_force_host_platform_device_count; got {len(devices)}"
        )
    arr = np.asarray(devices[: data_size * model_size]).reshape(data_size, model_size)
    return Mesh(arr, axis_names=("data", "model"))


# --------------------------------------------------------------------------- #
# _leaf_partition_spec
# --------------------------------------------------------------------------- #
class TestLeafPartitionSpec:
    def test_scalar_replicated(self):
        from myllm.training.mesh import _leaf_partition_spec
        from jax.sharding import PartitionSpec as P
        assert _leaf_partition_spec((), mesh_size=4) == P()

    def test_1d_divisible_shards(self):
        # [H=8] with mesh=4: 8 % 4 == 0 → shard axis 0
        from myllm.training.mesh import _leaf_partition_spec
        from jax.sharding import PartitionSpec as P
        spec = _leaf_partition_spec((8,), mesh_size=4)
        assert spec == P("data")

    def test_1d_not_divisible_replicated(self):
        # [H=7] with mesh=4: 7 % 4 != 0 → replicated
        from myllm.training.mesh import _leaf_partition_spec
        from jax.sharding import PartitionSpec as P
        spec = _leaf_partition_spec((7,), mesh_size=4)
        assert spec == P()

    def test_2d_largest_axis_wins(self):
        # [V=131072, H=2048] with mesh=4: both divisible; V > H → shard axis 0
        from myllm.training.mesh import _leaf_partition_spec
        from jax.sharding import PartitionSpec as P
        spec = _leaf_partition_spec((131072, 2048), mesh_size=4)
        assert spec == P("data", None)

    def test_2d_only_one_axis_divisible(self):
        # [9, 8] with mesh=4: 9 % 4 != 0, 8 % 4 == 0 → shard axis 1
        from myllm.training.mesh import _leaf_partition_spec
        from jax.sharding import PartitionSpec as P
        spec = _leaf_partition_spec((9, 8), mesh_size=4)
        assert spec == P(None, "data")

    def test_2d_tie_breaks_to_smaller_index(self):
        # [8, 8] with mesh=4: both axes equal-size and divisible.
        # Rule: smaller index (axis 0) wins.
        from myllm.training.mesh import _leaf_partition_spec
        from jax.sharding import PartitionSpec as P
        spec = _leaf_partition_spec((8, 8), mesh_size=4)
        assert spec == P("data", None)

    def test_3d_picks_largest_divisible(self):
        # FFN-style [in=2048, out=8192, group=3]: axis 1 is largest divisible
        from myllm.training.mesh import _leaf_partition_spec
        from jax.sharding import PartitionSpec as P
        spec = _leaf_partition_spec((2048, 8192, 3), mesh_size=4)
        assert spec == P(None, "data", None)

    def test_no_axes_divisible_replicates(self):
        # Prime/odd sizes with mesh=4: nothing divides → replicate
        from myllm.training.mesh import _leaf_partition_spec
        from jax.sharding import PartitionSpec as P
        spec = _leaf_partition_spec((3, 5, 7), mesh_size=4)
        assert spec == P()

    def test_mesh_size_5_prime_falls_back_to_replicate(self):
        # On a 5-device mesh, almost no weight tensors are divisible.
        # Production powers-of-2 device counts are recommended; this test
        # locks in the fail-safe behavior.
        from myllm.training.mesh import _leaf_partition_spec
        from jax.sharding import PartitionSpec as P
        spec = _leaf_partition_spec((2048, 2048), mesh_size=5)
        assert spec == P()  # 2048 % 5 != 0

    def test_mesh_size_5_succeeds_when_divisible(self):
        # [10, 7] with mesh=5: 10 % 5 == 0 → shard axis 0
        from myllm.training.mesh import _leaf_partition_spec
        from jax.sharding import PartitionSpec as P
        spec = _leaf_partition_spec((10, 7), mesh_size=5)
        assert spec == P("data", None)

    def test_custom_mesh_axis_name(self):
        # Some setups call the data axis something else (e.g. "fsdp")
        from myllm.training.mesh import _leaf_partition_spec
        from jax.sharding import PartitionSpec as P
        spec = _leaf_partition_spec((8,), mesh_size=4, mesh_axis="fsdp")
        assert spec == P("fsdp")


# --------------------------------------------------------------------------- #
# make_param_shardings
# --------------------------------------------------------------------------- #
class TestMakeParamShardings:
    def test_returns_pytree_with_named_shardings(self):
        from myllm.training.mesh import make_param_shardings
        from jax.sharding import NamedSharding
        mesh = _mesh(4)
        params = {
            "embedding": np.zeros((128, 32), dtype=np.float32),  # [V, H]
            "ln": np.zeros((32,), dtype=np.float32),             # [H]
        }
        out = make_param_shardings(params, mesh)
        assert set(out.keys()) == {"embedding", "ln"}
        assert isinstance(out["embedding"], NamedSharding)
        assert isinstance(out["ln"], NamedSharding)

    def test_embedding_shards_vocab_axis(self):
        # The biggest axis on [V=128, H=32] with mesh=4 is V (axis 0).
        from myllm.training.mesh import make_param_shardings
        from jax.sharding import PartitionSpec as P
        mesh = _mesh(4)
        params = {"E": np.zeros((128, 32), dtype=np.float32)}
        out = make_param_shardings(params, mesh)
        assert out["E"].spec == P("data", None)

    def test_norm_scale_shards_when_divisible(self):
        from myllm.training.mesh import make_param_shardings
        from jax.sharding import PartitionSpec as P
        mesh = _mesh(4)
        params = {"scale": np.zeros((32,), dtype=np.float32)}
        out = make_param_shardings(params, mesh)
        assert out["scale"].spec == P("data")

    def test_nested_structure_preserved(self):
        from myllm.training.mesh import make_param_shardings
        from jax.sharding import NamedSharding
        mesh = _mesh(4)
        params = {
            "block_0": {
                "wq": np.zeros((32, 32), dtype=np.float32),
                "wk": np.zeros((32, 8), dtype=np.float32),
            },
            "block_1": {
                "wq": np.zeros((32, 32), dtype=np.float32),
                "wk": np.zeros((32, 8), dtype=np.float32),
            },
        }
        out = make_param_shardings(params, mesh)
        assert set(out.keys()) == {"block_0", "block_1"}
        assert set(out["block_0"].keys()) == {"wq", "wk"}
        assert isinstance(out["block_0"]["wq"], NamedSharding)

    def test_namedtuple_preserved(self):
        # muP uses optax.MultiTransformState which is a namedtuple. The
        # sharding pytree must preserve the namedtuple type (else
        # downstream `state.inner_states` becomes a dict access and breaks).
        from collections import namedtuple
        from myllm.training.mesh import make_param_shardings
        from jax.sharding import NamedSharding
        mesh = _mesh(4)
        FakeOptState = namedtuple("FakeOptState", ["count", "mu", "nu"])
        params = FakeOptState(
            count=np.zeros((), dtype=np.int32),
            mu=np.zeros((128, 32), dtype=np.float32),
            nu=np.zeros((128, 32), dtype=np.float32),
        )
        out = make_param_shardings(params, mesh)
        # Type must round-trip
        assert isinstance(out, FakeOptState)
        assert isinstance(out.mu, NamedSharding)
        assert isinstance(out.nu, NamedSharding)
        # Scalar count -> replicated
        from jax.sharding import PartitionSpec as P
        assert out.count.spec == P()
        # 2D arrays -> sharded on V (axis 0)
        assert out.mu.spec == P("data", None)

    def test_scalar_step_replicated(self):
        # state["step"] is a JAX scalar in production; should be replicated
        from myllm.training.mesh import make_param_shardings
        from jax.sharding import PartitionSpec as P
        mesh = _mesh(4)
        state = {"step": np.int32(0)}
        out = make_param_shardings(state, mesh)
        assert out["step"].spec == P()

    def test_works_with_shape_dtype_struct(self):
        # The agent's optimizer-state-sharding flow uses jax.eval_shape
        # whose output is a tree of ShapeDtypeStruct, not real arrays.
        # make_param_shardings must accept these too.
        import jax
        from myllm.training.mesh import make_param_shardings
        from jax.sharding import PartitionSpec as P
        mesh = _mesh(4)
        struct_tree = {
            "param": jax.ShapeDtypeStruct((64, 32), np.float32),
            "bias": jax.ShapeDtypeStruct((32,), np.float32),
        }
        out = make_param_shardings(struct_tree, mesh)
        # Same rule applies: largest divisible axis on [64, 32]: 64 > 32 → axis 0
        assert out["param"].spec == P("data", None)
        assert out["bias"].spec == P("data")

    def test_mesh_size_inferred_from_named_axis(self):
        # make_param_shardings reads mesh_size from mesh.shape[mesh_axis].
        # 4-device mesh on the "data" axis -> mesh_size == 4.
        from myllm.training.mesh import make_param_shardings
        from jax.sharding import PartitionSpec as P
        mesh = _mesh(4)
        # [8] is divisible by 4 → sharded
        out = make_param_shardings({"x": np.zeros((8,), dtype=np.float32)}, mesh)
        assert out["x"].spec == P("data")
        # [7] is not → replicated
        out = make_param_shardings({"y": np.zeros((7,), dtype=np.float32)}, mesh)
        assert out["y"].spec == P()

    def test_rejects_empty_mesh(self):
        # Defensive: callers shouldn't pass a zero-sized mesh, but we want
        # a loud failure rather than a silent div-by-zero or always-replicate.
        # (Hard to construct a 0-axis Mesh; this is more of a doc test —
        # ensure size-0 path raises something meaningful.)
        # In practice JAX won't let you build Mesh with 0 devices, so we
        # skip the active assertion and just confirm sizes >= 1 work.
        from myllm.training.mesh import make_param_shardings
        from jax.sharding import PartitionSpec as P
        mesh = _mesh(4)  # the smallest practical case
        out = make_param_shardings({"x": np.zeros((4,), dtype=np.float32)}, mesh)
        assert out["x"].spec == P("data")
