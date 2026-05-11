"""Loss-spike watchdog.

Tracks the running mean and std of the training loss and flags spikes.
The training loop's response policy:
    - On a soft warning, log and continue.
    - On a hard spike, the loop should:
        1. Restore the most recent good checkpoint.
        2. Halve the LR (or apply the configured lr-recovery rule).
        3. Skip ahead in the data stream so the offending batch isn't replayed.
        4. Resume.

This module owns *detection*, not response. Response is the loop's job.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field


@dataclass
class LossSpikeWatchdog:
    """Welford running mean/std with sliding window; flags spikes by sigma."""

    window: int = 200
    soft_sigma: float = 3.0
    hard_sigma: float = 6.0
    min_observations: int = 50
    _losses: deque[float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.window < 10:
            raise ValueError("window too small (need >= 10)")
        if self.soft_sigma >= self.hard_sigma:
            raise ValueError("soft_sigma must be < hard_sigma")
        self._losses = deque(maxlen=self.window)

    def observe(self, loss: float) -> str:
        """Add a loss observation; return ``"ok"``, ``"soft"``, or ``"hard"``.

        Compares ``loss`` against the mean/std of the *prior* window before
        appending. Computing against the post-append window is wrong: a single
        outsized spike pulls the mean up by ``spike/n`` and inflates std,
        which can make a 1000× spike against a tight distribution look like
        only ~5σ instead of the true thousands-σ.
        """
        if not math.isfinite(loss):
            return "hard"
        verdict = "ok"
        n = len(self._losses)
        if n >= self.min_observations:
            mean = sum(self._losses) / n
            var = sum((x - mean) ** 2 for x in self._losses) / n
            std = math.sqrt(var)
            if std > 0.0:
                sigma = (loss - mean) / std
                if sigma >= self.hard_sigma:
                    verdict = "hard"
                elif sigma >= self.soft_sigma:
                    verdict = "soft"
        self._losses.append(loss)
        return verdict

    def reset(self) -> None:
        self._losses.clear()
