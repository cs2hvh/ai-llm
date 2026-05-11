"""Smoke tests for Phase 0 — runs without GPU."""
from __future__ import annotations

from pathlib import Path

import pytest

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_pilot_config_loads():
    from myllm.model import ModelConfig

    cfg = ModelConfig.from_yaml(CONFIGS / "pilot_250m.yaml")
    # v1: hidden 768, FFN 3072 (4x), 12 heads, 4 KV heads (3:1 GQA).
    assert cfg.layers == 16
    assert cfg.hidden_dim == 768
    assert cfg.ffn_dim == 3072
    assert cfg.num_heads == 12
    assert cfg.num_kv_heads == 4
    assert cfg.rope_base == 130000.0
    assert cfg.vocab_size == 131072  # v2: bumped from 128000 for Llama-3.x compat
    # Target ~252M params (250M class with 131k vocab).
    assert 200_000_000 <= cfg.param_count_estimate() <= 320_000_000


def test_base_config_loads():
    from myllm.model import ModelConfig

    cfg = ModelConfig.from_yaml(CONFIGS / "base_1b.yaml")
    # v1 matches Llama 3.2 1B exactly.
    assert cfg.layers == 16
    assert cfg.hidden_dim == 2048
    assert cfg.ffn_dim == 8192
    assert cfg.num_heads == 32
    assert cfg.num_kv_heads == 8
    assert cfg.rope_base == 500000.0
    assert cfg.scaled_init_for_residuals is True
    assert cfg.vocab_size == 131072  # v2: bumped from 128000 for Llama-3.x compat
    # ~1.25B params (131k vocab adds ~6M to the embedding).
    assert 1_100_000_000 <= cfg.param_count_estimate() <= 1_400_000_000


def test_head_dim_consistency_enforced():
    from pydantic import ValidationError

    from myllm.model import ModelConfig

    with pytest.raises(ValidationError):
        ModelConfig(
            name="bad",
            layers=2,
            hidden_dim=128,
            ffn_dim=256,
            num_heads=4,
            num_kv_heads=2,
            head_dim=64,  # 4*64=256 != 128
            vocab_size=1000,
            context_length=128,
        )


def test_gqa_divisibility_enforced():
    from pydantic import ValidationError

    from myllm.model import ModelConfig

    with pytest.raises(ValidationError):
        ModelConfig(
            name="bad",
            layers=2,
            hidden_dim=128,
            ffn_dim=256,
            num_heads=6,  # not divisible by 4
            num_kv_heads=4,
            head_dim=128 // 6,  # would also fail head_dim, but tests catch first
            vocab_size=1000,
            context_length=128,
        )
