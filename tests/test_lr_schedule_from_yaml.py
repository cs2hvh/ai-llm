"""Regression test for P0 audit extra: WSD schedule must read from yaml.

Before the 2026-05-12 fix, ``run_pretrain.init_model_and_optimizer``
hardcoded:
    warmup_steps = min(2000, total_steps // 10)
    decay_steps  = int(total_steps * 0.15)
    end_lr       = peak_lr * 0.1

The yaml's `lr_schedule.warmup_steps`, `decay_fraction`, `end_lr_ratio`
were SILENTLY IGNORED. Pilot's configured warmup=2000 worked by coincidence
but anything custom would have been lost.

The fix factored the schedule math into ``resolve_wsd_schedule_params``
so it can be tested without JAX/Keras dependencies.

2026-05-16 refactor: the helper moved from ``scripts/run_pretrain.py``
to ``myllm.training.state_init`` so library code (myllm.infer.predict)
can import it without the sys.path dance scripts use. Test now uses
the clean import path; the old AST-extraction shim is gone.
"""
from __future__ import annotations

import pytest

from myllm.training.state_init import resolve_wsd_schedule_params as resolve


# --------------------------------------------------------------------------- #
# Defaults (no yaml override)
# --------------------------------------------------------------------------- #
class TestDefaults:
    def test_warmup_caps_at_2000(self):
        """For very long runs, warmup defaults to 2000 not total//10."""
        s = resolve(peak_lr=1e-3, total_steps=100_000, lr_schedule_cfg=None)
        assert s["warmup_steps"] == 2000

    def test_warmup_uses_tenth_for_short_runs(self):
        s = resolve(peak_lr=1e-3, total_steps=10_000, lr_schedule_cfg=None)
        assert s["warmup_steps"] == 1000

    def test_decay_fraction_default_is_15_percent(self):
        s = resolve(peak_lr=1e-3, total_steps=10_000)
        assert s["decay_steps"] == 1500

    def test_end_lr_ratio_default_is_one_tenth(self):
        s = resolve(peak_lr=1e-3, total_steps=10_000)
        assert s["end_lr"] == pytest.approx(1e-4)


# --------------------------------------------------------------------------- #
# Yaml overrides — the actual P0 regression
# --------------------------------------------------------------------------- #
class TestYamlOverrides:
    def test_warmup_steps_from_yaml(self):
        """yaml.lr_schedule.warmup_steps must override the default."""
        s = resolve(
            peak_lr=1e-3, total_steps=10_000,
            lr_schedule_cfg={"warmup_steps": 250},
        )
        assert s["warmup_steps"] == 250, (
            "P0 regression: yaml warmup_steps must override default. "
            "If this is 1000, the hardcoded `min(2000, total // 10)` is still active."
        )

    def test_decay_fraction_from_yaml(self):
        s = resolve(
            peak_lr=1e-3, total_steps=10_000,
            lr_schedule_cfg={"decay_fraction": 0.25},
        )
        assert s["decay_steps"] == 2500

    def test_end_lr_ratio_from_yaml(self):
        s = resolve(
            peak_lr=1e-3, total_steps=10_000,
            lr_schedule_cfg={"end_lr_ratio": 0.03},
        )
        assert s["end_lr"] == pytest.approx(3e-5)

    def test_stable_steps_account_for_warmup_and_decay(self):
        """Stable = total - warmup - decay. Must hold across yaml overrides."""
        s = resolve(
            peak_lr=1e-3, total_steps=10_000,
            lr_schedule_cfg={"warmup_steps": 500, "decay_fraction": 0.2},
        )
        # decay_steps = 2000, warmup = 500 → stable = 7500
        assert s["stable_steps"] == 7500

    def test_all_yaml_fields_combined(self):
        s = resolve(
            peak_lr=5e-4, total_steps=10_000,
            lr_schedule_cfg={
                "warmup_steps": 200,
                "decay_fraction": 0.25,
                "end_lr_ratio": 0.05,
            },
        )
        assert s["warmup_steps"] == 200
        assert s["decay_steps"] == 2500
        assert s["stable_steps"] == 7300  # 10000 - 200 - 2500
        assert s["end_lr"] == pytest.approx(2.5e-5)
