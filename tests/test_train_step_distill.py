"""End-to-end integration test for distillation-aware train_step (R0).

Verifies that ``make_train_step`` correctly routes through:
  - plain CE when the batch has no teacher data (stable phase)
  - mixed CE+KL when the batch carries teacher_topk_logits/indices
    (decay phase)

Builds a real (tiny) TransformerLM and exercises a single optimizer step
through ``jax.jit``. This ensures the JIT-trace handles the optional
teacher fields correctly — the kind of "the loss compiled and ran" check
that pure-math loss tests can't give us.
"""
from __future__ import annotations

from pathlib import Path

import pytest

keras = pytest.importorskip("keras")
ops = keras.ops
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
optax = pytest.importorskip("optax")

from myllm.model import ModelConfig, build_model  # noqa: E402
from myllm.training.optimizer import OptimizerConfig, build_optimizer  # noqa: E402
from myllm.training.train_step import make_train_step  # noqa: E402

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _build_tiny_model_state():
    cfg = ModelConfig.from_yaml(CONFIGS / "tiny_test.yaml")
    model = build_model(cfg)
    opt = build_optimizer(OptimizerConfig(peak_lr=1.0e-3), lambda _: 1.0e-3)

    trainable = [v.value for v in model.trainable_variables]
    non_trainable = [v.value for v in model.non_trainable_variables]
    opt_state = opt.init(trainable)
    state = {
        "trainable_variables": trainable,
        "non_trainable_variables": non_trainable,
        "opt_state": opt_state,
        "step": jnp.array(0, dtype=jnp.int32),
        "lr_recovery_multiplier": jnp.array(1.0, dtype=jnp.float32),
    }
    return cfg, model, opt, state


def _make_batch_without_teacher(cfg, batch_size=2, seq_len=16):
    ids = jax.random.randint(
        jax.random.PRNGKey(0), (batch_size, seq_len), 0, cfg.vocab_size
    )
    labels = jax.random.randint(
        jax.random.PRNGKey(1), (batch_size, seq_len), 0, cfg.vocab_size
    )
    return {"input_ids": ids, "labels": labels}


def _make_batch_with_teacher(cfg, batch_size=2, seq_len=16, n_teachers=2, k=4):
    """Batch + synthetic top-K teacher logits/indices."""
    base = _make_batch_without_teacher(cfg, batch_size, seq_len)
    teacher_logits = jax.random.normal(
        jax.random.PRNGKey(2), (n_teachers, batch_size, seq_len, k)
    )
    # Random valid vocab indices (each in [0, V)).
    teacher_indices = jax.random.randint(
        jax.random.PRNGKey(3),
        (n_teachers, batch_size, seq_len, k),
        0, cfg.vocab_size,
    )
    return {
        **base,
        "teacher_topk_logits": teacher_logits,
        "teacher_topk_indices": teacher_indices,
    }


def test_train_step_pure_ce_when_no_teacher_data():
    """Train step with alpha=1.0 and no teacher data: behaves as standard
    CE-only training. KL metric is 0; CE = total loss.
    """
    cfg, model, opt, state = _build_tiny_model_state()
    step_fn = make_train_step(model, opt, distill_alpha=1.0)
    batch = _make_batch_without_teacher(cfg)
    _, metrics = step_fn(state, batch)
    # Loss is finite and positive.
    assert bool(jnp.isfinite(metrics["loss"]))
    assert float(metrics["loss"]) > 0.0
    # KL=0 (no teacher provided).
    assert float(metrics["kl"]) == 0.0
    # alpha=1 recorded.
    assert float(metrics["alpha"]) == pytest.approx(1.0)
    # ce + z_loss term should approximately equal loss (z_loss_coef·z_loss is tiny).
    assert float(metrics["ce"]) == pytest.approx(
        float(metrics["loss"]) - 1.0e-4 * float(metrics["z_loss"]),
        rel=1.0e-4,
    )


def test_train_step_mixed_loss_when_teacher_data_present():
    """Train step with alpha=0.3 and teacher data: distillation activates.
    KL > 0 in metrics; total ≈ 0.3·CE + 0.7·KL (ignoring z_loss).
    """
    cfg, model, opt, state = _build_tiny_model_state()
    step_fn = make_train_step(model, opt, distill_alpha=0.3)
    batch = _make_batch_with_teacher(cfg)
    _, metrics = step_fn(state, batch)
    # Loss is finite and positive.
    assert bool(jnp.isfinite(metrics["loss"]))
    assert float(metrics["loss"]) > 0.0
    # KL > 0 since synthetic teacher diverges from student.
    assert float(metrics["kl"]) > 1.0e-3, (
        f"expected positive KL with teacher data, got {float(metrics['kl'])}"
    )
    # alpha=0.3 recorded.
    assert float(metrics["alpha"]) == pytest.approx(0.3, rel=1.0e-4)
    # Mixed loss ≈ 0.3·(CE + z_loss_coef·z_loss) + 0.7·KL.
    expected = (
        0.3 * (float(metrics["ce"]) + 1.0e-4 * float(metrics["z_loss"]))
        + 0.7 * float(metrics["kl"])
    )
    assert float(metrics["loss"]) == pytest.approx(expected, rel=1.0e-4)


def test_train_step_alpha_one_with_teacher_data_collapses_to_ce():
    """alpha=1.0 + teacher data: KL is computed (observability) but loss
    is pure CE (no distillation weight).
    """
    cfg, model, opt, state = _build_tiny_model_state()
    step_fn = make_train_step(model, opt, distill_alpha=1.0)
    batch = _make_batch_with_teacher(cfg)
    _, metrics = step_fn(state, batch)
    # KL is still observed.
    assert float(metrics["kl"]) > 0.0
    # But loss is CE+z_loss only (no KL term).
    expected_ce_plus_z = float(metrics["ce"]) + 1.0e-4 * float(metrics["z_loss"])
    assert float(metrics["loss"]) == pytest.approx(expected_ce_plus_z, rel=1.0e-4)


def test_train_step_updates_state_step_counter():
    """Sanity: state["step"] increments by 1 each call."""
    cfg, model, opt, state = _build_tiny_model_state()
    step_fn = make_train_step(model, opt)
    batch = _make_batch_without_teacher(cfg)
    new_state, _ = step_fn(state, batch)
    assert int(new_state["step"]) == 1
    new_state2, _ = step_fn(new_state, batch)
    assert int(new_state2["step"]) == 2
