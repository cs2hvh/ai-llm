"""Forward-pass smoke tests for the TransformerLM. Skipped if Keras not installed."""
from __future__ import annotations

from pathlib import Path

import pytest

keras = pytest.importorskip("keras")
ops = keras.ops

from myllm.model import (  # noqa: E402
    ModelConfig,
    RMSNorm,
    SwiGLUFFN,
    TransformerLM,
    apply_rope,
    build_model,
    causal_mask,
    precompute_rope_cache,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture(scope="module")
def tiny_cfg() -> ModelConfig:
    return ModelConfig.from_yaml(CONFIGS / "tiny_test.yaml")


def test_rmsnorm_shape_and_scale():
    layer = RMSNorm(dim=16)
    layer.build((None, 8, 16))
    x = ops.ones((2, 8, 16))
    y = layer(x)
    assert tuple(ops.shape(y)) == (2, 8, 16)
    # ones input → variance 1 → output = weight (initialized to ones).
    assert float(ops.max(ops.abs(y - 1.0))) < 1e-5


def test_rope_cache_shape_and_unitary():
    cos, sin = precompute_rope_cache(head_dim=16, max_seq_len=32)
    assert tuple(ops.shape(cos)) == (32, 8)
    assert tuple(ops.shape(sin)) == (32, 8)
    # cos^2 + sin^2 == 1 everywhere.
    s2 = cos * cos + sin * sin
    assert float(ops.max(ops.abs(s2 - 1.0))) < 1e-5


def test_apply_rope_preserves_shape_and_norm():
    cos, sin = precompute_rope_cache(head_dim=16, max_seq_len=8)
    x = keras.random.normal((2, 8, 4, 16))
    y = apply_rope(x, cos, sin)
    assert tuple(ops.shape(y)) == tuple(ops.shape(x))
    # RoPE is a rotation → preserves L2 norm per token.
    nx = ops.sum(x * x, axis=-1)
    ny = ops.sum(y * y, axis=-1)
    assert float(ops.max(ops.abs(nx - ny))) < 1e-3


def test_swiglu_forward():
    layer = SwiGLUFFN(hidden_dim=32, ffn_dim=64)
    layer.build((None, 4, 32))
    y = layer(keras.random.normal((2, 4, 32)))
    assert tuple(ops.shape(y)) == (2, 4, 32)


def test_causal_mask_shape_and_triangular():
    m = causal_mask(seq_len=4)
    assert tuple(ops.shape(m)) == (1, 1, 4, 4)
    arr = ops.convert_to_numpy(m)
    # Upper triangle (i<j) is large negative.
    for i in range(4):
        for j in range(4):
            if j > i:
                assert arr[0, 0, i, j] < -1e8
            else:
                assert arr[0, 0, i, j] == 0.0


def test_transformer_forward_pass(tiny_cfg):
    model = build_model(tiny_cfg)
    ids = ops.cast(keras.random.uniform((2, 16), maxval=tiny_cfg.vocab_size), "int32")
    logits = model(ids)
    assert tuple(ops.shape(logits)) == (2, 16, tiny_cfg.vocab_size)
    # Logits are finite (no NaN/Inf at init).
    assert bool(ops.all(ops.isfinite(logits)))


def test_param_count_matches_estimate(tiny_cfg):
    model = build_model(tiny_cfg)
    actual = int(sum(int(ops.size(v)) for v in model.trainable_variables))
    estimated = tiny_cfg.param_count_estimate()
    # The estimate is rough (ignores norm scales, biases) — within 10%.
    assert abs(actual - estimated) / estimated < 0.10, (
        f"actual={actual:,} estimated={estimated:,}"
    )


# --------------------------------------------------------------------------- #
# Dossier-2026-05-11 R2 (intra-doc masking) + R3 (QK-norm) regression tests
# --------------------------------------------------------------------------- #
def _tiny_cfg_with(qk_norm: bool):
    """Return a fresh tiny ModelConfig with qk_norm overridden."""
    cfg = ModelConfig.from_yaml(CONFIGS / "tiny_test.yaml")
    return cfg.model_copy(update={"qk_norm": qk_norm})


def test_qk_norm_forward_pass_finite():
    """Building with qk_norm=True must not break forward pass."""
    cfg = _tiny_cfg_with(qk_norm=True)
    model = build_model(cfg)
    ids = ops.cast(keras.random.uniform((2, 16), maxval=cfg.vocab_size), "int32")
    logits = model(ids)
    assert tuple(ops.shape(logits)) == (2, 16, cfg.vocab_size)
    assert bool(ops.all(ops.isfinite(logits)))


def test_qk_norm_adds_extra_params():
    """qk_norm should add 2 extra RMSNorm scales per attention layer."""
    cfg_off = _tiny_cfg_with(qk_norm=False)
    cfg_on = _tiny_cfg_with(qk_norm=True)
    m_off = build_model(cfg_off)
    m_on = build_model(cfg_on)
    n_off = sum(int(ops.size(v)) for v in m_off.trainable_variables)
    n_on = sum(int(ops.size(v)) for v in m_on.trainable_variables)
    # Expect 2 (q_norm + k_norm) × head_dim × layers extra params.
    expected_delta = 2 * cfg_on.head_dim * cfg_on.layers
    assert n_on - n_off == expected_delta, (
        f"qk_norm param delta: got {n_on - n_off}, expected {expected_delta}"
    )


def test_segment_mask_blocks_cross_document_attention(tiny_cfg):
    """With segment_ids set, tokens in segment 1 must NOT see segment 0.

    Construct two batches with identical tokens at positions 2-3 but different
    tokens at positions 0-1. Mark positions [0,1] as segment 0 and [2,3] as
    segment 1. With doc-masking enabled, the model's output at positions 2-3
    must be identical across the two batches — proving that segment 0's
    content didn't leak into segment 1.
    """
    model = build_model(tiny_cfg)
    ids = ops.array(
        [
            [10, 20, 100, 200],   # batch 0
            [30, 40, 100, 200],   # batch 1: positions 0-1 differ; 2-3 identical
        ],
        dtype="int32",
    )
    segs = ops.array([[0, 0, 1, 1], [0, 0, 1, 1]], dtype="int32")

    logits = model(ids, segment_ids=segs)
    diff = float(ops.max(ops.abs(logits[0, 2:4, :] - logits[1, 2:4, :])))
    assert diff < 1.0e-4, (
        f"segment-1 outputs leaked cross-segment info: max diff {diff:.6g}"
    )


def test_segment_mask_differs_from_pure_causal(tiny_cfg):
    """Without segment_ids the same input should produce different output at
    positions 2-3 (because causal attention from those positions sees
    positions 0-1 too). Verifies the segment mask path is actually doing work.
    """
    model = build_model(tiny_cfg)
    ids = ops.array(
        [
            [10, 20, 100, 200],
            [30, 40, 100, 200],
        ],
        dtype="int32",
    )
    # Pure causal: tokens at pos 2-3 attend to pos 0-1, which differ → outputs differ.
    logits_causal = model(ids)
    diff_causal = float(
        ops.max(ops.abs(logits_causal[0, 2:4, :] - logits_causal[1, 2:4, :]))
    )
    assert diff_causal > 1.0e-3, (
        f"pure-causal outputs unexpectedly identical: max diff {diff_causal:.6g}"
    )


def test_segment_mask_with_qk_norm_works_together():
    """The two dossier P0 fixes must compose: qk_norm=True AND segment_ids
    set should still produce finite output and respect the segment boundary.
    """
    cfg = _tiny_cfg_with(qk_norm=True)
    model = build_model(cfg)
    ids = ops.array(
        [
            [10, 20, 100, 200],
            [30, 40, 100, 200],
        ],
        dtype="int32",
    )
    segs = ops.array([[0, 0, 1, 1], [0, 0, 1, 1]], dtype="int32")
    logits = model(ids, segment_ids=segs)
    assert bool(ops.all(ops.isfinite(logits)))
    diff = float(ops.max(ops.abs(logits[0, 2:4, :] - logits[1, 2:4, :])))
    assert diff < 1.0e-4, (
        f"segment mask broken with qk_norm enabled: max diff {diff:.6g}"
    )
