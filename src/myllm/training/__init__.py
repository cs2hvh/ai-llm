"""JAX mesh setup, train loop, optimizer, Orbax checkpointing.

Most modules in this package are JAX/Optax/Orbax-dependent and are imported
lazily so that simply importing ``myllm.training`` does not pull in JAX.
"""

from myllm.training.mesh import ShardingConfig
from myllm.training.schedule import cosine_with_warmup
from myllm.training.state import TrainState
from myllm.training.watchdog import LossSpikeWatchdog

__all__ = [
    "TrainState",
    "cosine_with_warmup",
    "LossSpikeWatchdog",
    "ShardingConfig",
]
