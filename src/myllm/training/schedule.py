"""Learning-rate schedules. Pure-Python; no backend dependency.

Two schedules supported:
    - cosine_with_warmup: warmup → cosine decay to ``end_lr_ratio * peak_lr``
    - wsd: Warmup-Stable-Decay (linear warmup → constant peak → linear decay).
      Recommended for pretraining because you don't commit to a final token
      budget upfront — any stable-phase checkpoint can be cooled in 10-15%
      of remaining compute, and high-quality data can be mixed only into
      the decay phase (Llama 3, OLMo 2, MiniCPM, SmolLM2 all use this trick).
"""
from __future__ import annotations

import math


def cosine_with_warmup(
    step: int,
    *,
    peak_lr: float,
    warmup_steps: int,
    total_steps: int,
    end_lr_ratio: float = 0.1,
) -> float:
    """Linear warmup then cosine decay.

    Args:
        step: current step (0-indexed).
        peak_lr: maximum LR reached at end of warmup.
        warmup_steps: linear ramp from 0 to ``peak_lr`` over this many steps.
        total_steps: end of decay; LR plateaus at ``peak_lr * end_lr_ratio`` after.
        end_lr_ratio: fraction of peak_lr at end of decay.

    Returns:
        LR for this step.
    """
    if peak_lr <= 0:
        raise ValueError("peak_lr must be positive")
    if warmup_steps < 0 or total_steps <= warmup_steps:
        raise ValueError(
            f"need 0 <= warmup_steps < total_steps, got {warmup_steps}, {total_steps}"
        )
    if not 0 <= end_lr_ratio <= 1:
        raise ValueError("end_lr_ratio must be in [0, 1]")

    if step < warmup_steps:
        return peak_lr * step / max(1, warmup_steps)
    if step >= total_steps:
        return peak_lr * end_lr_ratio
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak_lr * (end_lr_ratio + (1.0 - end_lr_ratio) * cos)


def wsd(
    step: int,
    *,
    peak_lr: float,
    warmup_steps: int,
    decay_steps: int,
    total_steps: int,
    end_lr_ratio: float = 0.1,
) -> float:
    """Warmup-Stable-Decay schedule.

    Phases:
        [0, warmup_steps):                          linear 0 -> peak_lr
        [warmup_steps, total_steps - decay_steps):  constant peak_lr
        [total_steps - decay_steps, total_steps):   linear peak_lr -> end_lr
        [total_steps, ∞):                           constant end_lr

    Args:
        step: current step (0-indexed).
        peak_lr: maximum LR reached at end of warmup, held through stable phase.
        warmup_steps: linear ramp from 0 to ``peak_lr``.
        decay_steps: length of the linear decay phase at the end.
        total_steps: end of training (final step). Must satisfy
            ``warmup_steps + decay_steps <= total_steps``.
        end_lr_ratio: fraction of peak_lr at end of decay.

    Returns:
        LR for this step.
    """
    if peak_lr <= 0:
        raise ValueError("peak_lr must be positive")
    if warmup_steps < 0 or decay_steps < 0:
        raise ValueError("warmup_steps and decay_steps must be non-negative")
    if warmup_steps + decay_steps > total_steps:
        raise ValueError(
            f"warmup ({warmup_steps}) + decay ({decay_steps}) "
            f"must be <= total_steps ({total_steps})"
        )
    if not 0 <= end_lr_ratio <= 1:
        raise ValueError("end_lr_ratio must be in [0, 1]")

    end_lr = peak_lr * end_lr_ratio
    decay_start = total_steps - decay_steps

    if step < warmup_steps:
        return peak_lr * step / max(1, warmup_steps)
    if step < decay_start:
        return peak_lr
    if step >= total_steps:
        return end_lr
    # Linear decay from peak_lr to end_lr over [decay_start, total_steps).
    progress = (step - decay_start) / max(1, decay_steps)
    return peak_lr + (end_lr - peak_lr) * progress
