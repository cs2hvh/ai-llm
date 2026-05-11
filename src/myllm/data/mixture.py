"""Token-weighted mixture sampler over named data sources.

Each source produces a stream of training examples (Documents pre-tokenization,
or token lists post-tokenization). The ``MixtureSampler`` interleaves them so
the long-run **token share** drawn from each source matches its target weight.

Why this matters (P0-6 from 2026-05-12 audit):
    The earlier implementation sampled one DOCUMENT per step, weighted by
    yaml share. But document length varies wildly between sources:
    FineWeb-Edu docs ~500 tokens, pg19 books ~30K tokens. A pg19 share of
    0.05 with that doc length picks tokens ~3× as fast as a FineWeb-Edu
    share of 0.31 — silently inverting the intended token mix.

    The yaml's ``share`` field has always been documented as a token share
    (per the dossier R4 / pretraining playbook). Fixing the implementation
    to match is a P0 correctness bug.

Algorithm: online deficit-driven reweighting.
    On each pick:
      - Compute deficit_i = max(0, target_share_i * total_emitted - emitted_i)
        where emitted_i is cumulative size (tokens or chars) from source i.
      - Sample source ∝ deficit (zero deficit → not picked unless all zero).
      - At t=0 with all deficits=0, fall back to original weights for the
        first few picks to bootstrap.
      - Update emitted_i ← emitted_i + measure(example).

    This is self-correcting: any source that gets ahead of its target
    share is sampled less until others catch up. In the long run, the
    observed token share converges to the target.

Backwards compat:
    - If ``measure_fn`` is not passed, MixtureSampler measures examples by
      ``len(example.text)`` for Document-like objects, ``len(example)`` for
      list/str/tuple, else 1 (one-example-per-pick legacy behavior).
    - The yielded type is still ``(source_name, example)`` — no API break.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, TypeVar

from myllm.utils.exceptions import DataPipelineError

T = TypeVar("T")


@dataclass(frozen=True)
class SourceWeight:
    name: str
    weight: float


def _default_measure(example: Any) -> int:
    """Default size measurement: tokens for list, chars for Document/str.

    For Documents (data.types.Document): len(text) — char count is a
    reasonable proxy for English/Latin token count (~4 chars/token avg).
    For lists of ints (post-tokenization): len() — actual token count.
    For anything else: 1 (one example per pick).
    """
    # Avoid hasattr-everywhere; just probe what we actually emit.
    text = getattr(example, "text", None)
    if isinstance(text, str):
        return max(1, len(text))
    if isinstance(example, (list, tuple, str)):
        return max(1, len(example))
    return 1


@dataclass
class MixtureSampler:
    """Interleave examples from multiple sources so token shares match weights.

    Args:
        sources: mapping from source name to an iterable of examples.
        weights: target token-share weights per source; need not sum to 1
            (they are normalised). Sources with weight 0 are dropped.
        seed: PRNG seed for reproducibility.
        on_exhaust: ``"stop"`` to terminate when any source is exhausted,
            ``"drop"`` to remove that source and continue with renormalised
            weights, ``"cycle"`` to restart that source from the beginning.
        measure_fn: callable ``(example) -> int`` returning the size of one
            example for token-share accounting. Defaults to
            ``_default_measure`` (chars for Document, tokens for list,
            1 for opaque). Pass a tokenizer-aware measure to get exact
            token shares when sampling from tokenized streams.
        bootstrap_steps: for the first N picks, fall back to original
            weights instead of deficit-driven (the deficit signal is too
            noisy at startup). Default 32 picks.
    """

    sources: dict[str, Iterable[T]]
    weights: list[SourceWeight]
    seed: int = 0
    on_exhaust: str = "drop"
    measure_fn: Callable[[Any], int] = field(default=_default_measure)
    bootstrap_steps: int = 32
    _rng: random.Random = field(init=False, repr=False)
    # Cumulative size emitted per source. Public so callers can introspect
    # (e.g. for telemetry / regression tests on token shares).
    emitted_per_source: dict[str, int] = field(init=False, default_factory=dict)

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
        emitted: dict[str, int] = {n: 0 for n in order}
        self.emitted_per_source = emitted  # expose for introspection

        # Pre-normalise target shares.
        total_weight = sum(weights.values())
        target_share = {n: weights[n] / total_weight for n in order}

        step = 0
        while iters:
            # Pick source: bootstrap-mode uses raw weights, steady-state uses
            # deficits (max(0, target * total - emitted)).
            total_emitted = sum(emitted[n] for n in order)
            if step < self.bootstrap_steps or total_emitted == 0:
                picks_weights = [weights[n] for n in order]
            else:
                deficits = [
                    max(0.0, target_share[n] * total_emitted - emitted[n])
                    for n in order
                ]
                if sum(deficits) > 0:
                    picks_weights = deficits
                else:
                    # All caught up — sample by raw weights.
                    picks_weights = [weights[n] for n in order]

            picked = self._rng.choices(order, weights=picks_weights, k=1)[0]
            try:
                ex = next(iters[picked])
                emitted[picked] += self.measure_fn(ex)
                step += 1
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
                del target_share[picked]
                # Recompute total + renormalise target_share over survivors.
                total_weight = sum(weights.values())
                if total_weight <= 0:
                    return
                target_share = {n: weights[n] / total_weight for n in weights}
                order.remove(picked)
