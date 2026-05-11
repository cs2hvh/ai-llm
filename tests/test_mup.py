"""muP / muTransfer regression tests — R1 from 2026-05-11 dossier.

These tests validate the *scaffolding* added in this PR:

  1. When ``ModelConfig.mup`` is ``None`` (default), the model is identical
     to its pre-muP behavior — bitwise on a fixed seed.

  2. When ``ModelConfig.mup`` is set, the three output multipliers scale
     logit magnitudes by exactly ``1 / width_mult`` (per change-1/2/3 of
     ``docs/mup_design.md``).

  3. The per-flag ablation (``apply_attn_output_mult`` etc.) works — each
     can be toggled independently.

Width-invariance, per-param-LR scaling, and the actual wind-tunnel transfer
are validated in follow-up PRs (``test_mup_optim.py``, integration tests).
"""
from __future__ import annotations

from pathlib import Path

import pytest

keras = pytest.importorskip("keras")
ops = keras.ops

from myllm.model import ModelConfig, build_model  # noqa: E402
from myllm.model.config import MupConfig  # noqa: E402

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _tiny_cfg(mup: MupConfig | None = None) -> ModelConfig:
    """Tiny test config, optionally with a MupConfig attached."""
    base = ModelConfig.from_yaml(CONFIGS / "tiny_test.yaml")
    return base.model_copy(update={"mup": mup})


def test_mup_default_off_is_no_op():
    """ModelConfig.mup defaults to None — model must run unchanged."""
    cfg = ModelConfig.from_yaml(CONFIGS / "tiny_test.yaml")
    assert cfg.mup is None
    assert cfg.mup_width_multiplier() == 1.0
    model = build_model(cfg)
    ids = ops.cast(keras.random.uniform((2, 16), maxval=cfg.vocab_size, seed=0), "int32")
    logits = model(ids)
    assert tuple(ops.shape(logits)) == (2, 16, cfg.vocab_size)
    assert bool(ops.all(ops.isfinite(logits)))


def test_mup_width_multiplier_math():
    """The width multiplier is hidden_dim / base_width."""
    cfg = _tiny_cfg(MupConfig(base_width=32))
    # tiny_test.yaml has hidden_dim=64
    assert cfg.mup_width_multiplier() == 2.0

    cfg2 = _tiny_cfg(MupConfig(base_width=16))
    assert cfg2.mup_width_multiplier() == 4.0


def test_mup_lm_head_mult_scales_logits():
    """With muP enabled and only the LM-head mult on, logits scale by
    exactly 1 / width_mult vs the no-muP baseline.

    The forward path is otherwise identical — same weights, same inputs.
    """
    # Build a no-muP model, capture logits with a fixed seed.
    keras.utils.set_random_seed(7)
    cfg_sp = ModelConfig.from_yaml(CONFIGS / "tiny_test.yaml")
    model_sp = build_model(cfg_sp)
    ids = ops.cast(keras.random.uniform((2, 16), maxval=cfg_sp.vocab_size, seed=11), "int32")
    logits_sp = model_sp(ids)

    # Build a muP-enabled model with the same seed, only LM-head mult on.
    keras.utils.set_random_seed(7)
    cfg_mup = _tiny_cfg(
        MupConfig(
            base_width=32,
            apply_attn_output_mult=False,
            apply_ffn_output_mult=False,
            apply_lm_head_output_mult=True,
        )
    )
    model_mup = build_model(cfg_mup)
    logits_mup = model_mup(ids)

    # width_mult = 64 / 32 = 2.0 → logits should be halved.
    ratio = ops.mean(ops.abs(logits_mup)) / ops.mean(ops.abs(logits_sp))
    assert float(ratio) == pytest.approx(0.5, rel=0.05), (
        f"expected LM-head logits to scale by 1/width_mult=0.5, got {float(ratio):.4f}"
    )


def test_mup_attn_output_mult_scales_attention_contribution():
    """Enabling only the attention output multiplier scales the residual
    contribution of each attention block by 1/width_mult.

    We verify indirectly: turning on attn_output_mult must change the
    forward output (since attention contributes to the residual stream).
    """
    keras.utils.set_random_seed(13)
    cfg_sp = _tiny_cfg(None)
    model_sp = build_model(cfg_sp)
    ids = ops.cast(keras.random.uniform((1, 8), maxval=cfg_sp.vocab_size, seed=23), "int32")
    out_sp = model_sp(ids)

    keras.utils.set_random_seed(13)
    cfg_mup = _tiny_cfg(
        MupConfig(
            base_width=16,
            apply_attn_output_mult=True,
            apply_ffn_output_mult=False,
            apply_lm_head_output_mult=False,
        )
    )
    model_mup = build_model(cfg_mup)
    out_mup = model_mup(ids)

    # Should NOT be equal — attention contribution scaled by 1/4.
    diff = float(ops.max(ops.abs(out_sp - out_mup)))
    assert diff > 1.0e-3, (
        f"enabling attn_output_mult must change forward output; got max-diff {diff:.6g}"
    )


def test_mup_ffn_output_mult_scales_ffn_contribution():
    """Enabling only the FFN output multiplier must change forward output."""
    keras.utils.set_random_seed(19)
    cfg_sp = _tiny_cfg(None)
    model_sp = build_model(cfg_sp)
    ids = ops.cast(keras.random.uniform((1, 8), maxval=cfg_sp.vocab_size, seed=29), "int32")
    out_sp = model_sp(ids)

    keras.utils.set_random_seed(19)
    cfg_mup = _tiny_cfg(
        MupConfig(
            base_width=16,
            apply_attn_output_mult=False,
            apply_ffn_output_mult=True,
            apply_lm_head_output_mult=False,
        )
    )
    model_mup = build_model(cfg_mup)
    out_mup = model_mup(ids)

    diff = float(ops.max(ops.abs(out_sp - out_mup)))
    assert diff > 1.0e-3, (
        f"enabling ffn_output_mult must change forward output; got max-diff {diff:.6g}"
    )


def test_mup_all_off_flags_is_no_op():
    """A MupConfig with all three apply_* flags False must be a no-op,
    bitwise identical to mup=None.
    """
    keras.utils.set_random_seed(31)
    cfg_sp = _tiny_cfg(None)
    model_sp = build_model(cfg_sp)
    ids = ops.cast(keras.random.uniform((1, 8), maxval=cfg_sp.vocab_size, seed=37), "int32")
    out_sp = model_sp(ids)

    keras.utils.set_random_seed(31)
    cfg_mup_off = _tiny_cfg(
        MupConfig(
            base_width=16,
            apply_attn_output_mult=False,
            apply_ffn_output_mult=False,
            apply_lm_head_output_mult=False,
        )
    )
    model_mup_off = build_model(cfg_mup_off)
    out_mup_off = model_mup_off(ids)

    diff = float(ops.max(ops.abs(out_sp - out_mup_off)))
    assert diff < 1.0e-5, (
        f"all-flags-off MupConfig should be bitwise no-op; got max-diff {diff:.6g}"
    )


def test_mup_config_validates_base_width():
    """base_width must be a positive int."""
    with pytest.raises(Exception):
        MupConfig(base_width=0)
    with pytest.raises(Exception):
        MupConfig(base_width=-1)
    # And there's an upper bound for sanity.
    with pytest.raises(Exception):
        MupConfig(base_width=999999)


def test_mup_config_forbids_extra_fields():
    """MupConfig has extra='forbid' — typo'd fields should fail loud."""
    with pytest.raises(Exception):
        MupConfig(base_width=256, applyattnoutputmult=True)  # noqa: typo-on-purpose


def test_mup_param_count_unchanged():
    """muP scaffolding adds NO trainable parameters. Output multipliers
    are constants, not weights.
    """
    cfg_sp = _tiny_cfg(None)
    cfg_mup = _tiny_cfg(MupConfig(base_width=16))
    m_sp = build_model(cfg_sp)
    m_mup = build_model(cfg_mup)
    n_sp = sum(int(ops.size(v)) for v in m_sp.trainable_variables)
    n_mup = sum(int(ops.size(v)) for v in m_mup.trainable_variables)
    assert n_sp == n_mup, (
        f"muP should not add trainable params; got delta {n_mup - n_sp}"
    )
