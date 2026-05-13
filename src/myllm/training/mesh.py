"""JAX device mesh + sharding helpers.

A 2D mesh ``("data", "model")`` covers both regimes we use:
    - Pilot 250M on 1 pod (8 H100): ``data=8, model=1``. Pure data parallel.
      Parameters replicated; inputs sharded along the batch axis.
    - Base 1B on 4 pods (32 GPU) with FSDP: ``data=32, model=1`` for plain
      FSDP, or ``data=4, model=8`` if combining DP across pods with FSDP
      within a pod. (Real FSDP also requires PartitionSpecs on weight
      tensors; planned for the next iteration.)

This module deliberately raises on device-count mismatch — silent
under-utilisation is harder to debug than a loud failure at startup.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ShardingConfig:
    data_parallel: int
    model_parallel: int = 1

    def __post_init__(self) -> None:
        if self.data_parallel < 1 or self.model_parallel < 1:
            raise ValueError("data_parallel and model_parallel must be >= 1")

    @property
    def total_devices(self) -> int:
        return self.data_parallel * self.model_parallel


def build_mesh_and_shardings(config: ShardingConfig) -> tuple[Any, Any, Any]:
    """Return ``(mesh, data_sharding, replicate_sharding)``.

    ``data_sharding`` splits a tensor along its first (batch) axis across
    the ``data`` mesh dim. ``replicate_sharding`` replicates a tensor on
    every device.
    """
    try:
        import jax  # noqa: F401
        import numpy as np
        from jax.sharding import Mesh, NamedSharding
        from jax.sharding import PartitionSpec as P
    except ImportError as e:
        raise ImportError("jax not installed; install jax[cuda12]") from e

    devices = jax.devices()
    if len(devices) != config.total_devices:
        raise RuntimeError(
            f"sharding expects {config.total_devices} devices "
            f"({config.data_parallel} data x {config.model_parallel} model), "
            f"jax.devices() returned {len(devices)}"
        )
    device_array = np.asarray(devices).reshape(
        config.data_parallel, config.model_parallel
    )
    mesh = Mesh(device_array, axis_names=("data", "model"))
    data_sharding = NamedSharding(mesh, P("data"))
    replicate_sharding = NamedSharding(mesh, P())
    return mesh, data_sharding, replicate_sharding


# Backwards-compat shim — the old name returned just the Mesh.
@dataclass(frozen=True)
class MeshConfig(ShardingConfig):
    """Alias kept for callers that imported the old name."""


def build_mesh(config: ShardingConfig) -> Any:
    """Return just the Mesh (legacy entry point)."""
    mesh, _, _ = build_mesh_and_shardings(config)
    return mesh


def shard_state(state: dict[str, Any], replicate_sharding: Any) -> dict[str, Any]:
    """Place every tensor in ``state`` onto every device (replicated).

    2026-05-12 re-audit P0 fix: this function used to hardcode only 4
    state keys (trainable_variables, non_trainable_variables, opt_state,
    step), dropping ``lr_recovery_multiplier`` and ``data_position`` and
    any other operational keys the loop relies on. Now generic: every
    key in ``state`` survives. Python scalars / dicts that ``device_put``
    can't handle pass through unchanged (e.g. integer ``step``).
    """
    try:
        import jax
    except ImportError as e:
        raise ImportError("jax not installed") from e

    def put_if_arraylike(x: Any) -> Any:
        # device_put accepts arrays + scalars. Python ints, strings,
        # nested dicts that aren't JAX-typed get returned unchanged so
        # the caller can keep tracking them.
        try:
            return jax.device_put(x, replicate_sharding)
        except (TypeError, ValueError):
            return x

    return {key: jax.tree.map(put_if_arraylike, value) for key, value in state.items()}


def shard_batch(batch: dict[str, Any], data_sharding: Any) -> dict[str, Any]:
    """Shard each tensor in ``batch`` along axis 0 across the ``data`` mesh dim."""
    try:
        import jax
    except ImportError as e:
        raise ImportError("jax not installed") from e
    return {k: jax.device_put(v, data_sharding) for k, v in batch.items()}


# --------------------------------------------------------------------------- #
# FSDP / ZeRO-3 sharding helpers (2026-05-13)
#
# These build on `build_mesh_and_shardings` and let the caller construct a
# PyTree of NamedShardings (one per leaf) that distributes parameters,
# optimizer state, and gradients across the ``data`` mesh axis.
#
# Why this exists: the existing ``shard_state`` replicates every leaf onto
# every device. At 1B params on 5x H200 with our config, that's ~20 GB of
# replicated optimizer state per device. FSDP/ZeRO-3 sharding cuts that to
# ~4 GB/device (5x), which is what unlocks larger micro-batch and higher
# MFU. The senior reviewer flagged DP-replicated state as our largest
# remaining infra blocker; this is the foundation of the fix.
#
# Design rule (per the FSDP plan agent, validated against MaxText/Levanter
# idioms):
#
#   For each leaf tensor with shape ``S = (s0, s1, ..., sN)``:
#     - If any axis ``si`` is divisible by the data-axis mesh size, shard
#       along the LARGEST such axis. Ties: smaller index wins (axis 0 bias).
#     - If no axis is divisible (or the leaf is a scalar), REPLICATE.
#     - Replication is safe correctness-wise; just costs memory.
#
# Limitation: on prime mesh sizes (e.g. 5x H200 single-pod), most weight
# tensors will fall back to replication. Use device counts of 2/4/8 for the
# memory win to actually materialise. (5x H200 testing is fine for parity
# canaries since correctness doesn't care which axis is sharded.)
# --------------------------------------------------------------------------- #
def _leaf_partition_spec(
    leaf_shape: tuple[int, ...],
    mesh_size: int,
    mesh_axis: str = "data",
) -> Any:
    """Return a ``PartitionSpec`` for one leaf, given its shape.

    Selects the largest axis divisible by ``mesh_size`` for sharding along
    ``mesh_axis``. Falls back to a fully-replicated spec when no axis is
    divisible (or for scalars).

    See module docstring for the design rule.
    """
    try:
        from jax.sharding import PartitionSpec as P
    except ImportError as e:
        raise ImportError("jax not installed; install jax[cuda12]") from e

    if not leaf_shape:
        return P()  # scalars (e.g. step, lr_recovery_multiplier)

    best_axis: int | None = None
    best_dim: int = -1
    for i, dim in enumerate(leaf_shape):
        # `dim > best_dim` (strict >) biases toward smaller index on ties.
        if dim % mesh_size == 0 and dim > best_dim:
            best_axis = i
            best_dim = dim

    if best_axis is None:
        return P()  # nothing divisible -> replicate

    spec_list: list[Any] = [None] * len(leaf_shape)
    spec_list[best_axis] = mesh_axis
    return P(*spec_list)


def make_param_shardings(
    params_pytree: Any,
    mesh: Any,
    mesh_axis: str = "data",
) -> Any:
    """Return a PyTree of ``NamedSharding`` with the same structure as ``params_pytree``.

    Walks ``params_pytree`` leaf-by-leaf and assigns each leaf a
    ``NamedSharding(mesh, _leaf_partition_spec(...))``. The result has
    EXACTLY the same tree structure as the input — including any
    namedtuple types (preserved by ``jax.tree.map``).

    Use:
        shardings = make_param_shardings(trainable_pytree, mesh)
        trainable_sharded = jax.tree.map(
            lambda x, s: jax.device_put(x, s), trainable_pytree, shardings,
        )

    The same helper works for optimizer state — its leaves have the same
    shapes as the params they track, so reusing this function preserves
    the parallel sharding structure. (Use ``make_optimizer_state_sharding``
    when you want the eval_shape -> sharding-of-init-output flow; see
    optimizer.py.)
    """
    try:
        import jax
        from jax.sharding import NamedSharding
    except ImportError as e:
        raise ImportError("jax not installed; install jax[cuda12]") from e

    mesh_size = int(mesh.shape[mesh_axis])
    if mesh_size < 1:
        raise ValueError(
            f"mesh axis {mesh_axis!r} has size {mesh_size}; expected >= 1"
        )

    def _make(leaf: Any) -> Any:
        # Tolerate both numpy/jax arrays and ShapeDtypeStruct (from
        # jax.eval_shape); both expose .shape.
        shape = getattr(leaf, "shape", None)
        if shape is None:
            # Plain Python scalar (e.g. int step). Treat as scalar.
            shape = ()
        return NamedSharding(mesh, _leaf_partition_spec(
            tuple(shape), mesh_size=mesh_size, mesh_axis=mesh_axis,
        ))

    return jax.tree.map(_make, params_pytree)
