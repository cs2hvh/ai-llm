"""MuonClip / QK-clip regression tests — D10 (post-review 2026-05-18).

Validates the standalone QK-clip rescale described in the Kimi K2 paper
(arXiv 2507.20534) and implemented in ``src/myllm/training/muonclip.py``.

  1. Below-threshold scores → W_q, W_k unchanged.
  2. Above-threshold scores → W_q, W_k rescaled symmetrically (alpha=0.5)
     such that the post-rescale max QK product is at most ``threshold``.
  3. Per the Kimi K2 algorithm: eta = threshold / max_score;
     wq *= eta**alpha, wk *= eta**(1-alpha). Math is exact at alpha=0.5.

No model required; this is a pure-functional test.
"""
from __future__ import annotations

import os
import pytest

os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from myllm.training.muonclip import apply_qk_clip  # noqa: E402


def test_below_threshold_is_identity():
    """When max_qk_score < threshold, W_q and W_k must pass through
    unchanged (jnp.where short-circuit; JIT-safe)."""
    wq = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
    wk = jnp.array([[5.0, 6.0], [7.0, 8.0]], dtype=jnp.float32)
    score = jnp.array(50.0, dtype=jnp.float32)  # below default t=100

    new_wq, new_wk = apply_qk_clip(wq, wk, score, threshold=100.0)
    assert jnp.allclose(new_wq, wq), "wq changed when below threshold"
    assert jnp.allclose(new_wk, wk), "wk changed when below threshold"


def test_above_threshold_rescales_symmetrically_at_alpha_0_5():
    """With max_score=200, threshold=100, alpha=0.5:
       eta = 100/200 = 0.5
       wq *= sqrt(0.5) = 0.7071
       wk *= sqrt(0.5) = 0.7071
    """
    wq = jnp.ones((2, 2), dtype=jnp.float32)
    wk = jnp.ones((2, 2), dtype=jnp.float32)
    score = jnp.array(200.0, dtype=jnp.float32)

    new_wq, new_wk = apply_qk_clip(wq, wk, score, threshold=100.0, alpha=0.5)
    expected = jnp.sqrt(jnp.array(0.5))
    assert jnp.allclose(new_wq, jnp.ones_like(wq) * expected, atol=1e-5)
    assert jnp.allclose(new_wk, jnp.ones_like(wk) * expected, atol=1e-5)


def test_above_threshold_asymmetric_alpha():
    """alpha=1.0 means: all rescale on W_q, none on W_k.
       eta = 100/400 = 0.25 → wq *= 0.25, wk *= 1.0
    """
    wq = jnp.ones((2, 2), dtype=jnp.float32) * 2.0
    wk = jnp.ones((2, 2), dtype=jnp.float32) * 3.0
    score = jnp.array(400.0, dtype=jnp.float32)

    new_wq, new_wk = apply_qk_clip(wq, wk, score, threshold=100.0, alpha=1.0)
    assert jnp.allclose(new_wq, jnp.ones_like(wq) * 0.5, atol=1e-5)
    assert jnp.allclose(new_wk, wk, atol=1e-7)


def test_rescale_bounds_post_clip_max_score():
    """Mathematical invariant: after the rescale, if we recompute the
    max QK product (under the same Q, K input scale), it must be at
    most `threshold`. This is the entire point of MuonClip.

    We mock this by computing q @ k.T with W_q, W_k applied to a unit
    input vector and asserting the bound.
    """
    # Set up so the unconstrained "max QK score" is 400.
    wq = jnp.array([[20.0]], dtype=jnp.float32)
    wk = jnp.array([[20.0]], dtype=jnp.float32)
    x = jnp.array([[1.0]], dtype=jnp.float32)
    pre = float((x @ wq @ wk.T @ x.T).squeeze())
    assert pre == pytest.approx(400.0)

    threshold = 100.0
    new_wq, new_wk = apply_qk_clip(
        wq, wk, jnp.array(pre, dtype=jnp.float32),
        threshold=threshold, alpha=0.5,
    )
    post = float((x @ new_wq @ new_wk.T @ x.T).squeeze())
    assert post <= threshold + 1e-4, (
        f"post-clip max QK score {post} exceeds threshold {threshold}"
    )


def test_jit_safe_no_python_branch_on_tracer():
    """The implementation must be JIT-friendly: jit-compiled and run
    without a TracerBoolConversionError when the trigger condition is
    a tracer-dependent scalar."""
    @jax.jit
    def f(wq, wk, score):
        return apply_qk_clip(wq, wk, score, threshold=100.0, alpha=0.5)

    wq = jnp.ones((2, 2), dtype=jnp.float32)
    wk = jnp.ones((2, 2), dtype=jnp.float32)

    # Below threshold — both should be unchanged under JIT.
    nwq, nwk = f(wq, wk, jnp.array(50.0))
    assert jnp.allclose(nwq, wq)

    # Above threshold — JIT must produce the same rescaled result.
    nwq, nwk = f(wq, wk, jnp.array(200.0))
    expected = jnp.sqrt(jnp.array(0.5))
    assert jnp.allclose(nwq, jnp.ones_like(wq) * expected, atol=1e-5)
