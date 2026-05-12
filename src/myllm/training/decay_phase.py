"""Decay-phase distillation activation — the runtime switch that turns
on teacher-guided training during the WSD decay phase.

R0 from the 2026-05-11 dossier; this module is the loop-side counterpart
to ``src/myllm/training/loss.py::distillation_mixed_loss`` and
``src/myllm/data/teacher_cache.py::MultiTeacherCacheReader``.

How activation works:

  - Stable phase (step < activation_step): the loop calls ``maybe_inject``
    and gets back the *unchanged* batch dict — no teacher fields, the
    train step's mixed-loss function falls through to pure CE+z-loss.
  - Decay phase (step >= activation_step): ``maybe_inject`` resolves the
    batch's absolute corpus positions, asks the multi-teacher cache reader
    for top-K logits/indices, and returns an augmented batch with
    ``teacher_topk_logits[T,B,S,K]`` + ``teacher_topk_indices[T,B,S,K]``
    fields. The train step's mixed-loss kicks in.

The split lets the loop carry one train_step function (compiled once,
with ``distill_alpha = 0.3``) across both phases. Stable phase doesn't
incur the KL cost because the loss function checks for the teacher
fields before computing KL.

Position tracking:

  Distillation requires knowing the *absolute corpus token position* of
  every position in the current batch — that's how the cache reader
  looks up the right cached top-K. The simplest pattern is a sequential
  counter: each batch consumes ``batch_size * seq_len`` positions starting
  where the last batch left off. ``SequentialCorpusPositions`` implements
  this. Resumability is a caller responsibility — pass the persisted
  start offset when constructing the tracker.

For data pipelines that read non-sequentially (e.g. shuffled shards), a
custom ``position_fn(state, batch) -> np.ndarray[int]`` can be passed
instead. The function must return one integer position per token in the
batch (i.e. shape ``(B*S,)``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from myllm.data.teacher_cache import MultiTeacherCacheReader
from myllm.utils import get_logger

log = get_logger(__name__)

# Type alias: (state_dict, batch_dict) -> 1-D int array of corpus positions.
PositionFn = Callable[[dict[str, Any], dict[str, Any]], Any]


class SequentialCorpusPositions:
    """Stateful position tracker for sequential corpus iteration.

    Each batch consumes ``B * S`` positions starting where the previous
    batch ended. Use this when the data iterator emits the corpus in
    order (typical for pretraining).

    The internal counter is incremented in ``__call__``. To resume, pass
    ``start_position`` from the persisted training state.
    """

    def __init__(self, start_position: int = 0):
        self._pos = int(start_position)

    @property
    def position(self) -> int:
        return self._pos

    def __call__(self, state: dict[str, Any], batch: dict[str, Any]):
        import numpy as np

        ids = batch["input_ids"]
        # Shape inference (handles both numpy arrays and JAX arrays).
        if hasattr(ids, "shape"):
            B, S = int(ids.shape[0]), int(ids.shape[1])
        else:
            B = len(ids)
            S = len(ids[0])
        n = B * S
        positions = np.arange(self._pos, self._pos + n, dtype=np.int64)
        self._pos += n
        return positions


@dataclass
class DecayPhaseActivation:
    """Glue between the training loop and the distillation infrastructure.

    Fields:
        activation_step:  step ``>= activation_step`` activates distillation
        total_steps:      total run length; used by ``current_alpha`` to
                          interpolate across the decay window. Defaults to
                          a sentinel that disables annealing (returns alpha_end).
        reader:           the multi-teacher cache reader; ``None`` disables
        position_fn:      ``(state, batch) -> np.ndarray[int]`` mapping each
                          batch position to its absolute corpus offset
        alpha_start:      CE weight at decay-phase start (default 0.7 per
                          external reviewer's recommendation; was 0.3 in v1)
        alpha_end:        CE weight at end of training (default 0.3)

    α-annealing (B8, 2026-05-12):
        At training step `s`, the loop calls ``current_alpha(s)`` and injects
        the result into the batch as ``batch["alpha"]``, which the train_step
        reads. Stable phase returns 1.0 (pure CE). Decay phase interpolates
        linearly from ``alpha_start`` at ``activation_step`` to ``alpha_end``
        at ``total_steps``.
    """

    activation_step: int
    reader: MultiTeacherCacheReader | None
    position_fn: PositionFn | None = None
    total_steps: int | None = None       # if None, alpha returns alpha_end during decay
    alpha_start: float = 0.7
    alpha_end: float = 0.3

    @classmethod
    def from_yaml(
        cls,
        yaml_path: str,
        total_steps: int,
        reader: MultiTeacherCacheReader | None,
        position_fn: PositionFn | None = None,
    ) -> "DecayPhaseActivation":
        """Build an activation policy from a `decay_phase_distillation.yaml`.

        Reads:
          - ``activation_fraction`` (float ∈ [0,1]) → ``activation_step``
          - ``alpha_schedule.start`` / ``.end`` if present (linear annealing
            across the decay window). Falls back to the legacy scalar
            ``alpha`` field for both endpoints if no schedule is set.
        """
        import yaml
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
        fraction = float(cfg.get("activation_fraction", 0.85))
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(
                f"activation_fraction must be in [0,1]; got {fraction}"
            )
        # Alpha schedule. If the yaml has an `alpha_schedule` block with
        # start/end, use that (B8 path, 2026-05-12). Otherwise fall back
        # to the legacy scalar `alpha` field (v1 behavior, constant alpha).
        sched = cfg.get("alpha_schedule") or {}
        legacy_alpha = float(cfg.get("alpha", 0.3))
        alpha_start = float(sched.get("start", legacy_alpha))
        alpha_end = float(sched.get("end", legacy_alpha))
        if sched.get("type") and sched["type"] != "linear":
            raise ValueError(
                f"only 'linear' alpha_schedule type supported; "
                f"got {sched['type']!r}"
            )
        return cls(
            activation_step=int(fraction * total_steps),
            reader=reader,
            position_fn=position_fn,
            total_steps=total_steps,
            alpha_start=alpha_start,
            alpha_end=alpha_end,
        )

    def current_alpha(self, step: int) -> float:
        """Return the CE-weight alpha for the given training step.

        Stable phase (step < activation_step):
            Returns 1.0 — pure cross-entropy, no distillation.

        Decay phase (step >= activation_step):
            Linear interpolation from ``alpha_start`` (at activation_step)
            to ``alpha_end`` (at total_steps). Clamped at alpha_end past the
            end of training.

        If ``total_steps`` is None (sentinel for "no schedule"), returns
        ``alpha_end`` for the whole decay phase (back-compat with v1
        constant-alpha behavior).
        """
        s = int(step)
        if s < self.activation_step:
            return 1.0
        if self.total_steps is None or self.total_steps <= self.activation_step:
            return float(self.alpha_end)
        # Linear interpolation over the decay window.
        decay_len = self.total_steps - self.activation_step
        progress = (s - self.activation_step) / max(1, decay_len)
        progress = max(0.0, min(1.0, progress))
        return float(self.alpha_start + (self.alpha_end - self.alpha_start) * progress)

    # ------------------------------------------------------------------- #
    # The loop's hook
    # ------------------------------------------------------------------- #
    def is_active(self, state: dict[str, Any]) -> bool:
        """True iff ``state["step"]`` is past the activation point AND a
        reader is configured."""
        if self.reader is None:
            return False
        return int(state.get("step", 0)) >= self.activation_step

    def maybe_inject(
        self, state: dict[str, Any], batch: dict[str, Any]
    ) -> dict[str, Any]:
        """Augment the batch with teacher top-K data if we're past activation.

        Stable phase: returns ``batch`` unchanged.
        Decay phase: returns a new dict with ``teacher_topk_logits[T,B,S,K]``
        and ``teacher_topk_indices[T,B,S,K]`` added.
        """
        if not self.is_active(state):
            return batch
        if self.position_fn is None:
            # The activation is configured but we have no way to map batch
            # tokens to corpus positions. Fail loud — silent fallback to CE
            # would mean we paid the cache-generation cost for nothing.
            raise RuntimeError(
                f"decay-phase distillation activated at step {state['step']} "
                "but no position_fn was provided. Pass SequentialCorpusPositions() "
                "or a custom callable to DecayPhaseActivation."
            )

        positions = self.position_fn(state, batch)
        n_positions = int(positions.shape[0]) if hasattr(positions, "shape") else len(positions)

        # Resolve top-K via the reader. Out-of-coverage positions raise.
        logits_flat, indices_flat = self.reader.get_topk(positions)
        # Shape: (T, n_positions, K) — reshape per batch dims.
        # batch["input_ids"] is [B, S]; we have B*S positions.
        ids = batch["input_ids"]
        if hasattr(ids, "shape"):
            B, S = int(ids.shape[0]), int(ids.shape[1])
        else:
            B, S = len(ids), len(ids[0])
        if B * S != n_positions:
            raise RuntimeError(
                f"position_fn returned {n_positions} positions but batch is "
                f"B*S = {B}*{S} = {B * S}. Position tracker out of sync."
            )
        T = logits_flat.shape[0]
        K = logits_flat.shape[-1]
        teacher_logits = logits_flat.reshape(T, B, S, K)
        teacher_indices = indices_flat.reshape(T, B, S, K)

        return {
            **batch,
            "teacher_topk_logits": teacher_logits,
            "teacher_topk_indices": teacher_indices,
        }
