"""Weighted mixture sampler over named data sources.

Each source produces a stream of training examples (typically token batches).
The ``MixtureSampler`` interleaves them so the long-run share of examples
drawn from each source matches its target ``weight``. Useful both at the
tokenizer-corpus stage and during pretraining for shard-level re-balancing.

Implementation: per-step sampling via ``random.choices`` with weights —
unbiased on each step, exact target rate in expectation. For deterministic
streaming we accept Stochastic Gradient Descent's tolerance for noise here;
strict round-robin or Gumbel-top-k would over-couple sources.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterable, Iterator, TypeVar

from myllm.utils.exceptions import DataPipelineError

T = TypeVar("T")


@dataclass(frozen=True)
class SourceWeight:
    name: str
    weight: float


@dataclass
class MixtureSampler:
    """Interleave examples from multiple sources at the configured rates.

    Args:
        sources: mapping from source name to an iterable of examples.
        weights: target sampling weights per source name; need not sum to 1
            (they are normalised). Sources with weight 0 are dropped.
        seed: PRNG seed for reproducibility.
        on_exhaust: ``"stop"`` to terminate when any source is exhausted,
            ``"drop"`` to remove that source and continue with renormalised
            weights, ``"cycle"`` to restart that source from the beginning.
    """

    sources: dict[str, Iterable[T]]
    weights: list[SourceWeight]
    seed: int = 0
    on_exhaust: str = "drop"
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.on_exhaust not in {"stop", "drop", "cycle"}:
            raise DataPipelineError(f"invalid on_exhaust: {self.on_exhaust}")
        names = {w.name for w in self.weights}
        missing = names - set(self.sources)
        if missing:
            raise DataPipelineError(f"weights reference unknown sources: {missing}")
        if any(w.weight < 0 for w in self.weights):
            raise DataPipelineError("weights must be non-negative")
        total = sum(w.weight for w in self.weights)
        if total <= 0:
            raise DataPipelineError("at least one source must have weight > 0")
        self._rng = random.Random(self.seed)

    def __iter__(self) -> Iterator[tuple[str, T]]:
        # Maintain mutable iterator state so we can drop or cycle on exhaustion.
        iters: dict[str, Iterator[T]] = {
            w.name: iter(self.sources[w.name])
            for w in self.weights
            if w.weight > 0
        }
        weights = {w.name: w.weight for w in self.weights if w.weight > 0}
        order = list(weights.keys())

        while iters:
            current_weights = [weights[n] for n in order]
            picked = self._rng.choices(order, weights=current_weights, k=1)[0]
            try:
                ex = next(iters[picked])
                yield picked, ex
            except StopIteration:
                if self.on_exhaust == "stop":
                    return
                if self.on_exhaust == "cycle":
                    iters[picked] = iter(self.sources[picked])
                    continue
                # "drop"
                del iters[picked]
                del weights[picked]
                order.remove(picked)
