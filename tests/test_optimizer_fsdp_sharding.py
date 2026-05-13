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
        # Agent-review fix B2 (2026-05-13): the previous version of this
        # test just flattened the tree and checked NamedSharding leaves
        # — it did NOT actually verify the namedtuple-preservation
        # property the test name claims. The B1-class bug (where
        # `state.inner_states` becomes a dict key access instead of a
        # namedtuple field) would have slipped through.
        #
        # This version walks the structure RECURSIVELY and asserts the
        # namedtuple type matches the real opt_state at every level.
        import jax
        from myllm.training.optimizer import make_optimizer_state_sharding

        mesh = _mesh(4)
        params = {
            "embed": np.zeros((128, 32), dtype=np.float32),
            "hidden_wq": np.zeros((32, 32), dtype=np.float32),
            "ln_norm": np.zeros((32,), dtype=np.float32),
        }
        opt = self._make_mup_optimizer(params, mup_width_mult=2.0)
        real_state = opt.init(params)
        shardings = make_optimizer_state_sharding(opt, params, mesh)

        # Top-level: should be optax.MultiTransformState (a namedtuple).
        # (No outer chain in _make_mup_optimizer; multi_transform is the
        # whole optimizer here.)
        assert type(shardings) is type(real_state), (
            f"top-level type mismatch: sharding={type(shardings).__name__} "
            f"real={type(real_state).__name__} — namedtuple NOT preserved"
        )

        # MultiTransformState must have a real .inner_states attribute
        # access, not a dict key access.
        assert hasattr(shardings, "inner_states"), (
            "shardings.inner_states is not an attribute — namedtuple was "
            "flattened. This is the B1-class bug class; would break "
            "opt_state updates at runtime."
        )
        # Same labels as real state
        assert set(shardings.inner_states.keys()) == set(real_state.inner_states.keys()), (
            "MultiTransformState.inner_states label set differs from real state"
        )

        # Walk each inner state — they're chain(adamw, scale) -> tuple of
        # states. Verify type identity at the nested level.
        for label in real_state.inner_states:
            real_inner = real_state.inner_states[label]
            shard_inner = shardings.inner_states[label]
            assert type(shard_inner) is type(real_inner), (
                f"inner state for label={label!r}: sharding type "
                f"{type(shard_inner).__name__} != real type "
                f"{type(real_inner).__name__}"
            )
            # The inner chain is (adamw_state, scale_state).
            # adamw_state is itself a tuple of (clip_or_scale_state, ScaleByAdamState).
            # Walk one level deeper to confirm namedtuple types persist.
            if isinstance(real_inner, tuple) and len(real_inner) > 0:
                for j, (r_elem, s_elem) in enumerate(zip(real_inner, shard_inner)):
                    assert type(s_elem) is type(r_elem), (
                        f"inner[{label}][{j}]: sharding type "
                        f"{type(s_elem).__name__} != real type "
                        f"{type(r_elem).__name__}"
                    )


class TestProductionChainStructure:
    """Locks the structure of the REAL optimizer chain that make_optimizer
    produces. The previous test class built a local multi_transform without
    the outer clip_by_global_norm; that's a real gap because the production
    chain has DIFFERENT top-level shape (a tuple of (EmptyState,
    MultiTransformState)) than a bare multi_transform.

    Agent-review-driven test (2026-05-13).
    """

    def test_production_chain_top_level_is_tuple_of_two_states(self):
        # make_optimizer with mup_width_mult > 1 returns
        #   chain(clip_by_global_norm, multi_transform(...))
        # whose init produces (clip_state, multi_state).
        import jax
        import optax
        from myllm.training.optimizer import (
            OptimizerConfig, build_optimizer,
            PARAM_GROUP_EMBEDDING, PARAM_GROUP_HIDDEN, PARAM_GROUP_NORM,
        )
        from myllm.training.optimizer import make_optimizer_state_sharding

        mesh = _mesh(4)
        # `build_optimizer` expects params + labels to be parallel lists
        # (matches `label_model_variables` output for a Keras model).
        params = [
            np.zeros((128, 32), dtype=np.float32),   # embedding
            np.zeros((32, 32), dtype=np.float32),    # hidden (wq)
            np.zeros((32,), dtype=np.float32),       # norm
        ]
        labels = [
            PARAM_GROUP_EMBEDDING,
            PARAM_GROUP_HIDDEN,
            PARAM_GROUP_NORM,
        ]
        # Constant LR for the test
        opt = build_optimizer(
            OptimizerConfig(peak_lr=1e-3),
            lr_fn=1e-3,
            mup_width_mult=2.0,
            param_labels=labels,
        )

        real_state = opt.init(params)
        shardings = make_optimizer_state_sharding(opt, params, mesh)

        # Top-level should be the same Python type as the real init output.
        # For a chain(...), optax returns a tuple-of-states.
        assert type(shardings) is type(real_state), (
            f"chain top-level type mismatch: sharding={type(shardings).__name__} "
            f"real={type(real_state).__name__}"
        )

        # Should have exactly 2 elements: (clip_state, multi_transform_state).
        # clip_by_global_norm has EmptyState (or similar trivial state).
        # multi_transform's state is MultiTransformState.
        assert len(real_state) == 2, f"unexpected real_state len: {len(real_state)}"
        assert len(shardings) == 2, f"sharding tuple len mismatch: {len(shardings)}"

        # Element 1 of real_state is MultiTransformState (namedtuple). The
        # sharding tree at position 1 must match.
        real_mt = real_state[1]
        shard_mt = shardings[1]
        assert type(shard_mt) is type(real_mt), (
            f"chain[1] type mismatch: sharding={type(shard_mt).__name__} "
            f"real={type(real_mt).__name__} — the MultiTransformState "
            f"namedtuple was NOT preserved inside the chain structure"
        )
        # Field access must work, not key access
        assert hasattr(shard_mt, "inner_states"), (
            "shardings[1].inner_states is not an attribute"
        )
        assert set(shard_mt.inner_states.keys()) == {
            PARAM_GROUP_EMBEDDING, PARAM_GROUP_NORM, PARAM_GROUP_HIDDEN
        }


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
