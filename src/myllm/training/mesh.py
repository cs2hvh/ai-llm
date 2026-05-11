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
    """Place every tensor in ``state`` onto every device (replicated)."""
    try:
        import jax
    except ImportError as e:
        raise ImportError("jax not installed") from e

    def put(x: Any) -> Any:
        return jax.device_put(x, replicate_sharding)

    return {
        "trainable_variables": jax.tree.map(put, state["trainable_variables"]),
        "non_trainable_variables": jax.tree.map(put, state["non_trainable_variables"]),
        "opt_state": jax.tree.map(put, state["opt_state"]),
        "step": state["step"],
    }


def shard_batch(batch: dict[str, Any], data_sharding: Any) -> dict[str, Any]:
    """Shard each tensor in ``batch`` along axis 0 across the ``data`` mesh dim."""
    try:
        import jax
    except ImportError as e:
        raise ImportError("jax not installed") from e
    return {k: jax.device_put(v, data_sharding) for k, v in batch.items()}
