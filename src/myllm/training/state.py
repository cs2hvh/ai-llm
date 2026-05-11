"""TrainState — the in-memory training-loop state.

Pure-Python dataclass. Backend tensors live inside ``params`` and
``opt_state``; we treat them opaquely. Anything serialised to a checkpoint
must be reconstructable from this dataclass plus the model config.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TrainState:
    """Snapshot of training loop state at a single step boundary."""

    step: int
    epoch: int
    params: Any  # PyTree of model parameters (backend-specific)
    opt_state: Any  # PyTree of optimizer state
    rng_key: Any  # JAX PRNG key (or None on TF)
    tokens_seen: int = 0
    last_loss: float | None = None

    def with_step(self, new_step: int, **kw: Any) -> "TrainState":
        """Return a new TrainState advanced to ``new_step``, overriding fields."""
        return TrainState(
            step=new_step,
            epoch=kw.get("epoch", self.epoch),
            params=kw.get("params", self.params),
            opt_state=kw.get("opt_state", self.opt_state),
            rng_key=kw.get("rng_key", self.rng_key),
            tokens_seen=kw.get("tokens_seen", self.tokens_seen),
            last_loss=kw.get("last_loss", self.last_loss),
        )
