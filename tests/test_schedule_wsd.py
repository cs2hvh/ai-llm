"""Tests for the WSD (Warmup-Stable-Decay) schedule."""
from __future__ import annotations

import pytest

from myllm.training.schedule import wsd


class TestWSD:
    def test_warmup_linear(self):
        # Linear ramp from 0 to peak_lr over warmup_steps.
        for s in [0, 25, 50, 75, 100]:
            lr = wsd(
                s,
                peak_lr=1.0,
                warmup_steps=100,
                decay_steps=150,
                total_steps=1000,
            )
            assert lr == pytest.approx(s / 100, rel=1e-6)

    def test_stable_phase_constant(self):
        # All steps in [warmup, total - decay) should equal peak_lr exactly.
        for s in [100, 200, 500, 700, 849]:
            lr = wsd(
                s,
                peak_lr=2.0,
                warmup_steps=100,
                decay_steps=150,
                total_steps=1000,
            )
            assert lr == pytest.approx(2.0)

    def test_decay_linear(self):
        # Linear decay from peak (2.0) to end (0.2) over decay_steps=150.
        # decay_start = 1000 - 150 = 850. end_lr = 0.2.
        lr_start = wsd(850, peak_lr=2.0, warmup_steps=100, decay_steps=150, total_steps=1000)
        lr_mid = wsd(925, peak_lr=2.0, warmup_steps=100, decay_steps=150, total_steps=1000)
        lr_end = wsd(1000, peak_lr=2.0, warmup_steps=100, decay_steps=150, total_steps=1000)
        assert lr_start == pytest.approx(2.0)
        assert lr_mid == pytest.approx(1.1, rel=1e-3)  # halfway: (2.0 + 0.2)/2
        assert lr_end == pytest.approx(0.2)

    def test_post_total_plateau(self):
        # Past total_steps, LR holds at end_lr.
        lr1 = wsd(1000, peak_lr=1.0, warmup_steps=100, decay_steps=150, total_steps=1000, end_lr_ratio=0.1)
        lr2 = wsd(2000, peak_lr=1.0, warmup_steps=100, decay_steps=150, total_steps=1000, end_lr_ratio=0.1)
        assert lr1 == lr2 == pytest.approx(0.1)

    def test_invalid_args(self):
        with pytest.raises(ValueError):
            wsd(0, peak_lr=-1, warmup_steps=10, decay_steps=10, total_steps=100)
        with pytest.raises(ValueError):
            # warmup + decay > total
            wsd(0, peak_lr=1, warmup_steps=60, decay_steps=60, total_steps=100)
        with pytest.raises(ValueError):
            wsd(0, peak_lr=1, warmup_steps=10, decay_steps=10, total_steps=100, end_lr_ratio=1.5)

    def test_zero_warmup_zero_decay_is_constant(self):
        # Edge case: no warmup, no decay → constant peak_lr throughout.
        for s in [0, 100, 500, 999]:
            lr = wsd(s, peak_lr=3.0, warmup_steps=0, decay_steps=0, total_steps=1000)
            assert lr == pytest.approx(3.0)
