"""JIT-compiled train step factory.

Returns ``train_step(state, batch) -> (state, metrics)`` suitable for use
inside ``jax.jit``.

State schema:
    trainable_variables:        sharded PyTree (Optax-compatible)
    non_trainable_variables:    PyTree (typically replicated)
    opt_state:                  PyTree from optimizer.init(...)
    step:                       int
    lr_recovery_multiplier:     float — halved on each watchdog hard-spike
                                recovery; multiplies the optimizer updates.
                                Set to 1.0 for normal training.

Batch schema:
    input_ids:                  ``[B, S]`` token ids.
    labels:                     ``[B, S]`` next-token targets.
    segment_ids:                optional ``[B, S]`` — when present, the
                                model receives it for intra-doc attention
                                masking (R2).
    teacher_topk_logits:        optional ``[T, B, S, K]`` — when present,
                                triggers the distillation loss path (R0).
                                Must be paired with teacher_topk_indices.
    teacher_topk_indices:       optional ``[T, B, S, K]`` int indices.

Keras 3 functional API:
    ``model.stateless_call(trainable_vars, non_trainable_vars, inputs)``
    returns ``(outputs, updated_non_trainable_vars)``.
"""
from __future__ import annotations

from typing import Any, Callable

from myllm.training.loss import distillation_mixed_loss


def make_train_step(
    model: Any,
    optimizer: Any,
    *,
    z_loss_coef: float = 1.0e-4,
    ignore_index: int | None = None,
    distill_alpha: float = 1.0,
    distill_temperature: float = 1.0,
    teacher_weights: tuple[float, ...] | None = None,
) -> Callable[[dict[str, Any], dict[str, Any]], tuple[dict[str, Any], dict[str, Any]]]:
    """Return a JIT-compiled ``(state, batch) -> (state, metrics)`` function.

    Distillation args (R0 from the 2026-05-11 dossier):
        distill_alpha:     CE weight in the mixed loss. ``1.0`` (default)
                           collapses to pure CE — matches pre-R0 behavior.
                           ``0.3`` is our locked decay-phase value per
                           ``docs/teacher_distillation_strategy.md``.
        distill_temperature: teacher softmax temperature.
        teacher_weights:   optional per-teacher weights for the ensemble.

    When the batch dict has no ``teacher_topk_logits`` key (or it is
    ``None``), the train step falls through to the plain CE+z-loss path
    regardless of ``distill_alpha`` — i.e. distillation only activates
    when teacher data is actually streamed into the batch. The intended
    usage is: stable-phase loop omits teacher data, decay-phase loop
    includes it; the same ``train_step`` function works for both.
    """
    try:
        import jax
        import optax
    except ImportError as e:
        raise ImportError("jax + optax required for train_step") from e

    def loss_fn(
        trainable: Any,
        non_trainable: Any,
        batch: dict[str, Any],
    ) -> tuple[Any, tuple[dict[str, Any], Any]]:
        logits, updated_non_trainable = model.stateless_call(
            trainable, non_trainable, batch["input_ids"]
        )
        # Teacher data is optional. When absent (stable phase) we get
        # plain CE+z-loss; when present (decay phase) we get the mixed
        # CE+KL loss with the configured alpha.
        teacher_logits = batch.get("teacher_topk_logits")
        teacher_indices = batch.get("teacher_topk_indices")
        loss, metrics = distillation_mixed_loss(
            logits,
            batch["labels"],
            teacher_logits,
            teacher_indices,
            alpha=distill_alpha,
            teacher_weights=teacher_weights,
            temperature=distill_temperature,
            ignore_index=ignore_index,
            z_loss_coef=z_loss_coef,
        )
        return loss, (metrics, updated_non_trainable)

    grad_fn = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)

    @jax.jit
    def train_step(
        state: dict[str, Any], batch: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        (loss, (metrics, new_non_trainable)), grads = grad_fn(
            state["trainable_variables"],
            state["non_trainable_variables"],
            batch,
        )
        updates, new_opt_state = optimizer.update(
            grads, state["opt_state"], state["trainable_variables"]
        )
        # Apply the watchdog recovery multiplier on top of the optimizer
        # schedule. lr_recovery_multiplier is 1.0 in normal training and gets
        # halved by the loop on each hard-spike recovery.
        mult = state["lr_recovery_multiplier"]
        updates = jax.tree.map(lambda u: u * mult, updates)
        new_trainable = optax.apply_updates(state["trainable_variables"], updates)
        new_state = {
            "trainable_variables": new_trainable,
            "non_trainable_variables": new_non_trainable,
            "opt_state": new_opt_state,
            "step": state["step"] + 1,
            "lr_recovery_multiplier": state["lr_recovery_multiplier"],
        }
        metrics_out = {"loss": loss, **metrics}
        return new_state, metrics_out

    return train_step
