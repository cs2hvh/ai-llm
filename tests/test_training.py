"""Unit tests for backend-independent training-module pieces."""
from __future__ import annotations

import math

import pytest

from myllm.training import LossSpikeWatchdog, TrainState, cosine_with_warmup


# --------------------------------------------------------------------------- #
# LR schedule
# --------------------------------------------------------------------------- #
class TestCosineWithWarmup:
    def test_warmup_linear(self):
        for s in range(0, 100, 10):
            lr = cosine_with_warmup(
                s, peak_lr=1.0, warmup_steps=100, total_steps=1000
            )
            assert lr == pytest.approx(s / 100, rel=1e-6)

    def test_peak_at_end_of_warmup(self):
        lr = cosine_with_warmup(100, peak_lr=1.0, warmup_steps=100, total_steps=1000)
        assert lr == pytest.approx(1.0, rel=1e-6)

    def test_decay_to_end_ratio(self):
        lr = cosine_with_warmup(
            1000, peak_lr=1.0, warmup_steps=100, total_steps=1000, end_lr_ratio=0.1
        )
        assert lr == pytest.approx(0.1, abs=1e-6)

    def test_post_decay_plateau(self):
        lr1 = cosine_with_warmup(
            1500, peak_lr=1.0, warmup_steps=100, total_steps=1000, end_lr_ratio=0.1
        )
        lr2 = cosine_with_warmup(
            5000, peak_lr=1.0, warmup_steps=100, total_steps=1000, end_lr_ratio=0.1
        )
        assert lr1 == lr2 == pytest.approx(0.1)

    def test_invalid_args(self):
        with pytest.raises(ValueError):
            cosine_with_warmup(0, peak_lr=-1, warmup_steps=10, total_steps=100)
        with pytest.raises(ValueError):
            cosine_with_warmup(0, peak_lr=1, warmup_steps=100, total_steps=50)


# --------------------------------------------------------------------------- #
# Watchdog
# --------------------------------------------------------------------------- #
class TestLossSpikeWatchdog:
    def test_quiet_during_warmup(self):
        wd = LossSpikeWatchdog(window=100, min_observations=50)
        for _ in range(40):
            assert wd.observe(2.0) == "ok"

    def test_no_spike_on_steady_loss(self):
        wd = LossSpikeWatchdog(min_observations=50)
        for _ in range(100):
            wd.observe(2.0 + 0.001)
        # No verdict will be 'soft' or 'hard' on near-flat data.
        for _ in range(20):
            v = wd.observe(2.0 + 0.001)
            assert v == "ok"

    def test_hard_spike_on_inf(self):
        wd = LossSpikeWatchdog()
        assert wd.observe(math.inf) == "hard"
        assert wd.observe(math.nan) == "hard"

    def test_hard_spike_on_extreme(self):
        wd = LossSpikeWatchdog(min_observations=50, soft_sigma=3.0, hard_sigma=6.0)
        # 100 stable obs around 2.0 ± 0.05.
        import random

        rng = random.Random(0)
        for _ in range(100):
            wd.observe(2.0 + rng.gauss(0, 0.05))
        # 20-sigma jump.
        v = wd.observe(50.0)
        assert v == "hard"

    def test_invalid_window(self):
        with pytest.raises(ValueError):
            LossSpikeWatchdog(window=5)
        with pytest.raises(ValueError):
            LossSpikeWatchdog(soft_sigma=5.0, hard_sigma=3.0)


# --------------------------------------------------------------------------- #
# TrainState
# --------------------------------------------------------------------------- #
class TestTrainState:
    def test_with_step_returns_new_state(self):
        s = TrainState(step=0, epoch=0, params="P", opt_state="O", rng_key=None)
        s2 = s.with_step(10, last_loss=1.5)
        assert s2.step == 10
        assert s2.last_loss == 1.5
        assert s.step == 0  # original unchanged
