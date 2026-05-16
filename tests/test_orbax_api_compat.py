"""Orbax API compatibility smoke test (Round A6, 2026-05-16).

Background
----------
Between Orbax 0.6 and 0.7 the restore-path semantics changed in two
non-backwards-compatible ways that bit us hard during pilot post-hoc
inference (commits 13d6126 -> 3be12de -> ca1c40b on 2026-05-14):

  1. ``ArrayRestoreArgs.__init__`` removed its ``shape=`` kwarg.
  2. 0-d scalar leaves (saved as 0-d tensorstore arrays) now require
     ``ArrayRestoreArgs(sharding=...)`` to deserialize; bare
     ``RestoreArgs()`` is rejected with "sharding ... Got None".

The user-facing fix shipped, but the *contract* between our code and
Orbax is now load-bearing enough that we want to fail loudly in CI if
any of the kwargs / call patterns we depend on change in a future bump.

This test exercises the *kwargs* path, not the full mesh-aware
deserialization (that's covered by tests/test_checkpoint_reshard.py).
It runs on CPU under any backend and has no GPU / multi-device deps.

When to update
--------------
If a future Orbax/JAX bump is intentional, refresh the pins in
pyproject.toml AND update the assertions here so the next reviewer can
see that "yes, the API surface drift was reviewed and re-locked."
"""
from __future__ import annotations

import inspect
import os
import tempfile
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest


# --------------------------------------------------------------------------- #
# Version + import surface
# --------------------------------------------------------------------------- #
class TestOrbaxJaxVersionsPinned:
    """The pyproject.toml pins are documented in the test so any future
    bump is a deliberate two-file change rather than a silent drift."""

    EXPECTED_JAX = "0.4.38"
    EXPECTED_ORBAX = "0.7.0"

    def test_jax_version_matches_pin(self):
        import jax
        assert jax.__version__ == self.EXPECTED_JAX, (
            f"jax {jax.__version__} != pinned {self.EXPECTED_JAX}. "
            f"If intentional, update pyproject.toml AND this test."
        )

    def test_orbax_version_matches_pin(self):
        import orbax.checkpoint as ocp
        assert ocp.__version__ == self.EXPECTED_ORBAX, (
            f"orbax {ocp.__version__} != pinned {self.EXPECTED_ORBAX}. "
            f"If intentional, update pyproject.toml AND this test, then "
            f"re-run tests/test_checkpoint_reshard.py to verify the G6 "
            f"reshard path still works against the new Orbax."
        )


# --------------------------------------------------------------------------- #
# ArrayRestoreArgs kwarg surface — the two failure modes that bit us
# --------------------------------------------------------------------------- #
class TestArrayRestoreArgsKwargs:
    """Pin the kwargs we pass to ArrayRestoreArgs."""

    def test_sharding_kwarg_accepted(self):
        # G6 path requires this. Verify it's still in the signature.
        import orbax.checkpoint as ocp
        sig = inspect.signature(ocp.ArrayRestoreArgs.__init__)
        assert "sharding" in sig.parameters, (
            "ArrayRestoreArgs lost the `sharding` kwarg — G6 cross-mesh "
            "restore is broken until checkpoint.restore() is updated."
        )

    def test_shape_kwarg_not_present(self):
        # Pre-2026-05-14 we passed shape= and it crashed. The Orbax 0.7
        # signature should NOT have it. If a future Orbax re-adds it, we
        # want to know — could mean a different API path is now preferred.
        import orbax.checkpoint as ocp
        sig = inspect.signature(ocp.ArrayRestoreArgs.__init__)
        assert "shape" not in sig.parameters, (
            "ArrayRestoreArgs regained a `shape` kwarg; revisit "
            "checkpoint.restore() — the 13d6126 fix may need to be "
            "re-evaluated against the new API."
        )

    def test_pytreecheckpointer_constructible(self):
        # Our checkpoint module does PyTreeCheckpointer() at module-load
        # time. This is the cheapest possible test that the import + ctor
        # path hasn't broken.
        import orbax.checkpoint as ocp
        ckpt = ocp.PyTreeCheckpointer()
        assert ckpt is not None


# --------------------------------------------------------------------------- #
# End-to-end save -> restore smoke
# --------------------------------------------------------------------------- #
def _tiny_state(data_position: int = 1_000_000):
    """Same shape as the production state pytree, minus the model weights.

    Verifies the leaf-types we care about survive save+restore:
      - 2-D float32 array (param-like)
      - 1-D float32 array (bias-like)
      - 0-D int32 scalar (step counter)
      - 0-D float32 scalar (lr_recovery_multiplier)
      - numpy int64 scalar (data_position)

    NOTE: ``data_position`` defaults to 1M to keep value < 2^31. Larger
    values (the production case post-fix 9f442f7 is up to ~5B) DO survive
    the legacy save/restore path, but get TRUNCATED on the per-leaf
    ArrayRestoreArgs path under JAX's default x32 mode (Orbax calls
    jax.device_put under the sharding, which respects JAX_ENABLE_X64).
    The int64-survival contract is therefore tested separately in
    test_data_position_int64_survives_legacy_path; the per-leaf-args
    path is only used by cross-mesh INFERENCE (generate.py,
    eval_checkpoint.py) where data_position is irrelevant, so the
    truncation is not a production correctness issue.
    """
    import jax.numpy as jnp
    return {
        "trainable_variables": [
            jnp.arange(64, dtype=jnp.float32).reshape(8, 8),
            jnp.arange(8, dtype=jnp.float32),
        ],
        "step": jnp.array(42, dtype=jnp.int32),
        "lr_recovery_multiplier": jnp.array(0.5, dtype=jnp.float32),
        "data_position": np.array(data_position, dtype=np.int64),
    }


def _all_close(a, b):
    """Walk two pytrees and assert leaf-by-leaf equality."""
    import jax
    leaves_a, def_a = jax.tree.flatten(a)
    leaves_b, def_b = jax.tree.flatten(b)
    assert def_a == def_b
    for i, (la, lb) in enumerate(zip(leaves_a, leaves_b)):
        arr_a = np.asarray(la)
        arr_b = np.asarray(lb)
        assert arr_a.shape == arr_b.shape, f"leaf {i} shape"
        assert arr_a.dtype == arr_b.dtype, f"leaf {i} dtype"
        assert np.array_equal(arr_a, arr_b), f"leaf {i} value"


class TestOrbaxSaveRestoreSmoke:
    """The three call patterns our checkpoint.py uses."""

    def test_save_then_restore_legacy_no_template(self):
        # The simplest call: ckpt.save(path, state); ckpt.restore(path).
        # Used when restoring on the same mesh layout with no template
        # (most-recent checkpoint, just-read-the-state case).
        import orbax.checkpoint as ocp
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir).resolve() / "state"
            ckpt = ocp.PyTreeCheckpointer()
            original = _tiny_state()
            ckpt.save(target, original)
            restored = ckpt.restore(target)
            _all_close(original, restored)

    def test_save_then_restore_with_template(self):
        # B1 fix path: pass item=template to preserve namedtuple types.
        # Our test state is a plain dict so structure is trivial, but
        # this confirms the kwarg is accepted + values round-trip.
        import orbax.checkpoint as ocp
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir).resolve() / "state"
            ckpt = ocp.PyTreeCheckpointer()
            original = _tiny_state()
            ckpt.save(target, original)
            restored = ckpt.restore(target, item=original)
            _all_close(original, restored)

    def test_save_then_restore_with_per_leaf_restore_args(self):
        # G6 path: build ArrayRestoreArgs per leaf via jax.tree.map.
        # This is the path that broke 3 times during pilot post-hoc
        # inference. Every leaf — including 0-d scalars — must get an
        # ArrayRestoreArgs with sharding=.
        import jax
        import orbax.checkpoint as ocp
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir).resolve() / "state"
            ckpt = ocp.PyTreeCheckpointer()
            original = _tiny_state(data_position=1_000_000)
            ckpt.save(target, original)

            sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
            restore_args = jax.tree.map(
                lambda _leaf: ocp.ArrayRestoreArgs(sharding=sharding),
                original,
            )
            restored = ckpt.restore(
                target, item=original, restore_args=restore_args,
            )
            assert np.array_equal(
                np.asarray(original["trainable_variables"][0]),
                np.asarray(restored["trainable_variables"][0]),
            )
            assert int(restored["step"]) == 42
            assert float(restored["lr_recovery_multiplier"]) == pytest.approx(0.5)
            assert int(restored["data_position"]) == 1_000_000

    def test_per_leaf_restore_args_demotes_int64_above_2_31(self):
        # KNOWN LIMITATION, documented as a test: under JAX x32 mode
        # (default), the per-leaf ArrayRestoreArgs path truncates int64
        # leaves to int32. Values < 2^31 round-trip fine; values above
        # overflow to negative.
        #
        # Production training-resume DOESN'T hit this — it uses the
        # legacy template-only restore (no sharding kwarg) which
        # preserves int64. The per-leaf path is only used by cross-mesh
        # INFERENCE (generate.py, eval_checkpoint.py) where data_position
        # is not consumed.
        #
        # If a future Orbax / JAX bump fixes this, the test will fail
        # and we should celebrate by deleting it. Until then we want it
        # to fail loudly if the truncation pattern changes (e.g., raises
        # instead of silently overflowing).
        import jax
        import orbax.checkpoint as ocp
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir).resolve() / "state"
            ckpt = ocp.PyTreeCheckpointer()
            big_val = 3_000_000_000  # > 2^31
            original = {"data_position": np.array(big_val, dtype=np.int64)}
            ckpt.save(target, original)

            sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
            restore_args = {
                "data_position": ocp.ArrayRestoreArgs(sharding=sharding),
            }
            restored = ckpt.restore(
                target, item=original, restore_args=restore_args,
            )
            # Silent int32 wrap. If this assertion ever fails, JAX/Orbax
            # may have started honoring int64 here — re-check whether we
            # can simplify our checkpoint.py.
            assert int(restored["data_position"]) == big_val - 2**32, (
                f"Expected x32-mode wraparound of {big_val}; got "
                f"{int(restored['data_position'])}. JAX or Orbax may "
                f"have changed int64 handling — review checkpoint.py."
            )

    def test_data_position_int64_survives_legacy_path(self):
        # Regression guard for the int32 overflow class. The production
        # training-resume path uses the LEGACY (template-only, no
        # sharding) restore. It must preserve int64 for values > 2^31.
        #
        # Compare: test_per_leaf_restore_args_demotes_int64_above_2_31
        # demonstrates that the OTHER restore path (with sharding) does
        # NOT preserve int64. The two paths have different dtype
        # contracts and that's load-bearing for our setup.
        import orbax.checkpoint as ocp
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir).resolve() / "state"
            ckpt = ocp.PyTreeCheckpointer()
            big_pos = 5_000_000_000  # > 2^31
            state = {"data_position": np.array(big_pos, dtype=np.int64)}
            ckpt.save(target, state)
            # Legacy path: no template, no sharding. Should preserve
            # dtype + value.
            restored = ckpt.restore(target)
            assert int(restored["data_position"]) == big_pos
            assert np.asarray(restored["data_position"]).dtype in (
                np.int64, np.uint64,
            )
