"""Integration test: train_step with use_chunked_ce=True vs False.

Wires together the new TransformerLM ``return_loss_inputs=True`` path with
the new ``chunked_cross_entropy_with_z_loss`` loss function via the
``use_chunked_ce`` flag in ``make_train_step``. The loss produced by the
chunked path must match the full-logit path within numerical tolerance
(1e-4 in float32; bf16 production runs would be ~1e-3).

Why this test exists: the senior reviewer (2026-05-12) flagged the full
[B,S,V] logit + one-hot label path as the dominant H200 OOM bottleneck.
Chunked CE addresses it by streaming the vocab. We need to be sure the
two paths produce *numerically equivalent* losses + gradients, otherwise
any later regression hides behind ~5-10% MFU variance and goes undetected.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# Force-import keras before model so backend selection is locked.
import keras  # noqa: F401, E402

from myllm.model.config import ModelConfig  # noqa: E402
from myllm.model.transformer import build_model  # noqa: E402
from myllm.training.train_step import make_train_step  # noqa: E402


def _tiny_cfg(vocab_size: int = 64) -> ModelConfig:
    return ModelConfig(
        name="tiny_chunked_ce_test",
        arch="llama_decoder",
        layers=2,
        hidden_dim=32,
        ffn_dim=64,
        num_heads=4,
        num_kv_heads=2,
        head_dim=8,
        vocab_size=vocab_size,
        tie_embeddings=True,
        context_length=16,
        position="rope",
        rope_base=10000.0,
        norm="rmsnorm",
        norm_eps=1e-5,
        activation="swiglu",
        init_std=0.02,
        scaled_init_for_residuals=False,
        z_loss_coef=1e-4,
        qk_norm=False,
    )


def _initial_state(model, optimizer):
    # Unwrap keras.Variable -> .value (raw JAX/Numpy array) so optax can init.
    # Matches the pattern in tests/test_train_step_distill.py.
    import jax.numpy as jnp
    trainable = [v.value for v in model.trainable_variables]
    non_trainable = [v.value for v in model.non_trainable_variables]
    return {
        "trainable_variables": trainable,
        "non_trainable_variables": non_trainable,
        "opt_state": optimizer.init(trainable),
        "step": jnp.array(0, dtype=jnp.int32),
        "lr_recovery_multiplier": jnp.array(1.0, dtype=jnp.float32),
    }


def _make_batch(seed: int, vocab: int, B: int = 1, S: int = 4):
    rng = np.random.default_rng(seed)
    return {
        "input_ids": rng.integers(0, vocab, size=(B, S)).astype(np.int32),
        "labels": rng.integers(0, vocab, size=(B, S)).astype(np.int32),
    }


class TestChunkedCEViaTrainStep:
    def test_return_loss_inputs_reconstructs_full_logits(self):
        # Sanity: model(ids, return_loss_inputs=True) yields the same
        # logits via matmul reconstruction as model(ids) yields directly.
        from keras import ops
        model = build_model(_tiny_cfg(vocab_size=64))
        ids = ops.convert_to_tensor(np.array([[1, 2, 3, 4]], dtype=np.int32))

        full_logits = model(ids)
        hidden, lm_w, mult = model(ids, return_loss_inputs=True)
        reconstructed = ops.matmul(hidden, ops.transpose(lm_w)) * mult

        err = float(ops.max(ops.abs(reconstructed - full_logits)))
        assert err < 1e-6, f"reconstruction error {err} too large"

    def test_chunked_ce_loss_matches_full_ce_loss(self):
        # End-to-end via train_step: identical batches, identical model
        # weights, identical optimizer — only the loss path differs.
        # The loss reported by the chunked path must match the full path.
        try:
            import optax
        except ImportError:
            pytest.skip("optax not installed")

        vocab = 64  # divisible by num_chunks=4
        cfg = _tiny_cfg(vocab_size=vocab)
        model = build_model(cfg)
        optimizer = optax.adamw(learning_rate=1e-3)
        batch = _make_batch(seed=42, vocab=vocab)

        # Build both train_steps over the SAME model. Each gets its own
        # freshly-built state — model trainable_variables are still the
        # same initialised values, but the state dicts don't share refs.
        step_full = make_train_step(
            model, optimizer, z_loss_coef=1e-4, use_chunked_ce=False
        )
        step_chunked = make_train_step(
            model, optimizer, z_loss_coef=1e-4,
            use_chunked_ce=True, chunked_ce_num_chunks=4,
        )

        state_a = _initial_state(model, optimizer)
        state_b = _initial_state(model, optimizer)

        _, metrics_full = step_full(state_a, batch)
        _, metrics_chunked = step_chunked(state_b, batch)

        ce_full = float(metrics_full["ce"])
        ce_chunked = float(metrics_chunked["ce"])
        z_full = float(metrics_full["z_loss"])
        z_chunked = float(metrics_chunked["z_loss"])

        assert np.isclose(ce_full, ce_chunked, atol=1e-4), \
            f"CE mismatch: full={ce_full} chunked={ce_chunked}"
        assert np.isclose(z_full, z_chunked, atol=1e-3), \
            f"z_loss mismatch: full={z_full} chunked={z_chunked}"
        # alpha + kl pad fields exist on both paths.
        assert "kl" in metrics_chunked
        assert "alpha" in metrics_chunked
        assert float(metrics_chunked["kl"]) == 0.0
        assert float(metrics_chunked["alpha"]) == 1.0

    def test_chunked_ce_falls_back_to_full_when_teacher_present(self):
        # When a teacher batch is provided, chunked path defers to the
        # full-logit distillation path. Verify the teacher-augmented
        # train_step still runs and produces a finite loss.
        try:
            import optax
        except ImportError:
            pytest.skip("optax not installed")

        vocab = 64
        cfg = _tiny_cfg(vocab_size=vocab)
        model = build_model(cfg)
        optimizer = optax.adamw(learning_rate=1e-3)
        state = _initial_state(model, optimizer)
        batch = _make_batch(seed=43, vocab=vocab)

        # Inject a fake teacher top-K.
        rng = np.random.default_rng(7)
        K = 4
        T = 1
        B, S = batch["input_ids"].shape
        batch["teacher_topk_logits"] = rng.standard_normal((T, B, S, K)).astype(np.float32)
        batch["teacher_topk_indices"] = rng.integers(0, vocab, size=(T, B, S, K)).astype(np.int32)

        step_chunked = make_train_step(
            model, optimizer, z_loss_coef=1e-4,
            distill_alpha=0.5, use_chunked_ce=True, chunked_ce_num_chunks=4,
        )
        new_state, metrics = step_chunked(state, batch)
        loss = float(metrics["ce"]) + float(metrics["z_loss"]) * 1e-4
        assert np.isfinite(loss), f"non-finite loss on teacher-present chunked path"
        assert "kl" in metrics, "kl key missing in metrics when teacher present"
