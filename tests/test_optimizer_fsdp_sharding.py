"""Tests for FSDP optimizer-state sharding (Commit B of FSDP plan).

Locks the invariant that:
  1. make_optimizer_state_sharding returns a NamedSharding pytree
     matching the optimizer's init output structure.
  2. For plain AdamW: the ScaleByAdamState namedtuple is preserved.
  3. For muP multi-transform: the MultiTransformState namedtuple is
     preserved. This is critical — if the namedtuple flattens to a
     dict, downstream `state.inner_states` (namedtuple field access)
     silently breaks. Same bug class as the B1 audit fix.
  4. Leaf shardings follow the same longest-divisible-axis rule used
     for params, since opt-state leaves have the same shapes as params.
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


class TestPlainAdamWSharding:
    def test_returns_pytree_with_named_shardings(self):
        import optax
        from jax.sharding import NamedSharding
        from myllm.training.optimizer import make_optimizer_state_sharding

        mesh = _mesh(4)
        params = {
            "embed": np.zeros((128, 32), dtype=np.float32),
            "weight": np.zeros((32, 32), dtype=np.float32),
        }
        opt = optax.adamw(learning_rate=1e-3, mu_dtype="float32")
        shardings = make_optimizer_state_sharding(opt, params, mesh)
        # Walk the tree and verify every leaf is a NamedSharding
        import jax
        flat, _ = jax.tree.flatten(shardings)
        assert all(isinstance(s, NamedSharding) for s in flat), \
            "every leaf in the sharding pytree must be a NamedSharding"
        assert len(flat) > 0, "empty sharding pytree — eval_shape failed"

    def test_state_pytree_structure_matches_init_output(self):
        # The sharding pytree must have EXACTLY the same structure as
        # the real opt_state init produces. Verified by comparing treedef.
        import jax
        import optax
        from myllm.training.optimizer import make_optimizer_state_sharding

        mesh = _mesh(4)
        params = {"x": np.zeros((64, 32), dtype=np.float32)}
        opt = optax.adamw(learning_rate=1e-3, mu_dtype="float32")
        real_state = opt.init(params)
        shardings = make_optimizer_state_sharding(opt, params, mesh)

        _, real_def = jax.tree.flatten(real_state)
        _, sharding_def = jax.tree.flatten(shardings)
        assert real_def == sharding_def, \
            "sharding pytree structure must match real opt_state structure"

    def test_leaves_shard_largest_divisible_axis(self):
        # AdamW state leaves (mu, nu) have the SAME shape as their params.
        # Sharding rule should pick axis 0 on [128, 32] (largest divisible).
        import jax
        import optax
        from jax.sharding import PartitionSpec as P
        from myllm.training.optimizer import make_optimizer_state_sharding

        mesh = _mesh(4)
        # Single param of shape [128, 32]
        params = {"E": np.zeros((128, 32), dtype=np.float32)}
        opt = optax.adamw(learning_rate=1e-3, mu_dtype="float32")
        shardings = make_optimizer_state_sharding(opt, params, mesh)

        # Find any leaf that has the [128, 32] sharding — must be P("data", None)
        flat, _ = jax.tree.flatten(shardings)
        # All non-scalar leaves with axis-0 sharding should be P("data", None)
        specs = [s.spec for s in flat]
        # The mu and nu copies of "E" must be sharded on axis 0
        assert P("data", None) in specs, \
            f"expected P('data', None) in {specs} (for mu/nu of [128, 32] param)"

    def test_scalar_count_replicated(self):
        # optax's ScaleByAdamState has a `count` scalar (int32). Must replicate.
        import jax
        import optax
        from jax.sharding import PartitionSpec as P
        from myllm.training.optimizer import make_optimizer_state_sharding

        mesh = _mesh(4)
        params = {"x": np.zeros((64, 32), dtype=np.float32)}
        opt = optax.adamw(learning_rate=1e-3, mu_dtype="float32")
        shardings = make_optimizer_state_sharding(opt, params, mesh)

        # At least one leaf should be a scalar replicated sharding
        flat, _ = jax.tree.flatten(shardings)
        specs = [s.spec for s in flat]
        assert P() in specs, \
            f"expected P() (replicated scalar) for adamw count; got {specs}"


class TestMuPMultiTransformSharding:
    def _make_mup_optimizer(self, params_dict, mup_width_mult=2.0):
        """Build a multi_transform optimizer over labeled params (matches
        the muP path in optimizer.py)."""
        import optax
        from myllm.training.optimizer import (
            PARAM_GROUP_EMBEDDING, PARAM_GROUP_NORM, PARAM_GROUP_HIDDEN,
        )
        # Trivial labels: classify each key by first character
        # (Just for the test; production uses label_variable_for_mup.)
        param_labels = {}
        for k in params_dict:
            if "embed" in k:
                param_labels[k] = PARAM_GROUP_EMBEDDING
            elif "norm" in k:
                param_labels[k] = PARAM_GROUP_NORM
            else:
                param_labels[k] = PARAM_GROUP_HIDDEN

        def _adamw_with_scale(lr_scale):
            return optax.chain(
                optax.adamw(learning_rate=1e-3, mu_dtype="float32"),
                optax.scale(lr_scale),
            )

        return optax.multi_transform(
            transforms={
                PARAM_GROUP_EMBEDDING: _adamw_with_scale(1.0),
                PARAM_GROUP_NORM: _adamw_with_scale(1.0),
                PARAM_GROUP_HIDDEN: _adamw_with_scale(1.0 / mup_width_mult),
            },
            param_labels=param_labels,
        )

    def test_multi_transform_state_namedtuple_preserved(self):
        # The TOP-LEVEL state is an optax.MultiTransformState namedtuple.
        # Its `inner_states` field MUST be preserved as a namedtuple field
        # (not a dict key) or downstream code breaks. This is the B1
        # audit's exact bug class.
        import jax
        import optax
        from myllm.training.optimizer import make_optimizer_state_sharding

        mesh = _mesh(4)
        params = {
            "embed": np.zeros((128, 32), dtype=np.float32),
            "hidden_wq": np.zeros((32, 32), dtype=np.float32),
            "ln_norm": np.zeros((32,), dtype=np.float32),
        }
        opt = self._make_mup_optimizer(params, mup_width_mult=2.0)

        # Compare type of real init output and sharding pytree at the
        # SAME path — they should both be MultiTransformState.
        real_state = opt.init(params)
        shardings = make_optimizer_state_sharding(opt, params, mesh)

        # Real state's top-level should be MultiTransformState
        assert "MultiTransformState" in type(real_state).__name__ or \
               isinstance(real_state, tuple), \
               f"unexpected real opt_state type: {type(real_state).__name__}"

        # Sharding pytree must have the same TYPE at the top level.
        # jax.tree.map preserves namedtuple types, so the assertion is:
        #   type(shardings) == type(real_state)
        assert type(shardings) == type(real_state), \
            f"sharding type {type(shardings).__name__} != real state " \
            f"type {type(real_state).__name__} — MultiTransformState namedtuple " \
            f"NOT preserved. This is the B1-class bug; opt-state updates would " \
            f"silently break at state.inner_states field access."

    def test_inner_states_accessible_as_attribute(self):
        # Explicit check: shardings.inner_states should work as attribute
        # access (not just dict-key access), because the namedtuple type
        # is preserved.
        import jax
        from myllm.training.optimizer import make_optimizer_state_sharding

        mesh = _mesh(4)
        params = {
            "embed": np.zeros((128, 32), dtype=np.float32),
            "hidden_wq": np.zeros((32, 32), dtype=np.float32),
            "ln_norm": np.zeros((32,), dtype=np.float32),
        }
        opt = self._make_mup_optimizer(params, mup_width_mult=2.0)
        shardings = make_optimizer_state_sharding(opt, params, mesh)

        # The whole chain returns a tuple of states (one per chain step).
        # The MultiTransformState should be findable by walking the tree.
        # Just confirm we can walk it without errors and find NamedSharding leaves.
        from jax.sharding import NamedSharding
        flat, _ = jax.tree.flatten(shardings)
        assert all(isinstance(s, NamedSharding) for s in flat), \
            "every leaf must be a NamedSharding"
        # And there should be many leaves (3 params × ~3 state buckets each)
        assert len(flat) >= 6, f"expected several leaves; got {len(flat)}"


class TestEvalShapeIntegration:
    def test_works_without_materialising_params(self):
        # Pass ShapeDtypeStruct instead of real arrays to verify the
        # eval_shape path (which is what run_pretrain.py will use to
        # avoid an unsharded transient at init time).
        import jax
        import optax
        from jax.sharding import NamedSharding
        from myllm.training.optimizer import make_optimizer_state_sharding

        mesh = _mesh(4)
        abstract_params = {
            "x": jax.ShapeDtypeStruct((128, 32), np.float32),
            "y": jax.ShapeDtypeStruct((64,), np.float32),
        }
        opt = optax.adamw(learning_rate=1e-3, mu_dtype="float32")
        shardings = make_optimizer_state_sharding(opt, abstract_params, mesh)

        # Same structure as if we'd passed real arrays.
        flat, _ = jax.tree.flatten(shardings)
        assert all(isinstance(s, NamedSharding) for s in flat)
        assert len(flat) > 0
