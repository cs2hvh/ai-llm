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
        # P0-2 fix (2026-05-12 audit): thread segment_ids into the model
        # call so the attention layer can build a same-document causal
        # mask. Without this, packed documents attend across boundaries
        # despite the model+layers code "supporting" segment_ids — the
        # data path simply wasn't passing it. The model's call signature
        # accepts `segment_ids=None`, which preserves back-compat for
        # synthetic-data batches that don't carry the field.
        segment_ids = batch.get("segment_ids")
        loss_mask = batch.get("loss_mask")
        call_kwargs = {}
        if segment_ids is not None:
            call_kwargs["segment_ids"] = segment_ids
        logits, updated_non_trainable = model.stateless_call(
            trainable, non_trainable, batch["input_ids"], **call_kwargs
        )
        # Teacher data is optional. When absent (stable phase) we get
        # plain CE+z-loss; when present (decay phase) we get the mixed
        # CE+KL loss with the configured alpha.
        teacher_logits = batch.get("teacher_topk_logits")
        teacher_indices = batch.get("teacher_topk_indices")
        # B8 fix (2026-05-12): alpha is now dynamic per step. The loop
        # computes alpha = decay_phase.current_alpha(step) and injects
        # it into the batch as a JAX scalar. If the batch doesn't carry
        # one (stable phase, synthetic data), fall back to the factory's
        # static default (which is 1.0 = pure CE).
        alpha = batch.get("alpha", distill_alpha)
        loss, metrics = distillation_mixed_loss(
            logits,
            batch["labels"],
            teacher_logits,
            teacher_indices,
            alpha=alpha,
            teacher_weights=teacher_weights,
            temperature=distill_temperature,
            ignore_index=ignore_index,
            z_loss_coef=z_loss_coef,
            loss_mask=loss_mask,
        )
        return loss, (metrics, updated_non_trainable)

    grad_fn = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)

    @jax.jit
    def train_step(
        state: dict[str, Any], batch: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        import jax.numpy as jnp

        (loss, (metrics, new_non_trainable)), grads = grad_fn(
            state["trainable_variables"],
            state["non_trainable_variables"],
            batch,
        )

        # Atomic NaN-skip (2026-05-12 audit P0-1 fix).
        #
        # If loss or any gradient is non-finite, we want the optimizer update
        # to be a no-op: params unchanged, optimizer state (m, v, step
        # counter) unchanged, non_trainable_variables unchanged. The 2026-05-11
        # version of this code only zeroed gradients, which is NOT sufficient:
        # AdamW's decoupled weight decay applies `lr * wd * params` regardless
        # of gradient magnitude, and the optimizer's internal step counter
        # (used for bias correction) advances inside `optimizer.update()`.
        # So a "skipped" batch with the old code would still drift params and
        # advance bias-correction denominators.
        #
        # The fix: always compute the candidate new state (params + opt_state +
        # non_trainable), then `jnp.where` on each leaf to pick the old state
        # when the batch was bad. This is atomic — no half-update gets through.
        #
        # The data-step counter (`state["step"]`) DOES advance even on a
        # skipped batch, so we don't get stuck replaying the same bad batch
        # forever. We track skip count via the `nan_skipped` metric so the
        # loop can log + alarm on it.
        #
        # Note on `non_trainable_variables` in this codebase: it contains
        # RoPE cos/sin tables (read-only constants). RMSNorm has no running
        # mean/var (unlike BatchNorm). So the per-leaf where on
        # non_trainable is a safe no-op today, but the pattern would also
        # correctly revert running stats if someone later adds BatchNorm.
        loss_finite = jnp.isfinite(loss)
        grads_finite = jax.tree.reduce(
            lambda a, b: a & b,
            jax.tree.map(lambda g: jnp.all(jnp.isfinite(g)), grads),
            jnp.array(True),
        )
        step_ok = loss_finite & grads_finite

        # Build candidate new state assuming the batch is good.
        # We feed `grads` (not zeroed grads) into the optimizer so that the
        # candidate state reflects what a normal update WOULD do; whether
        # that gets accepted is decided by the `step_ok` where below.
        updates, candidate_opt_state = optimizer.update(
            grads, state["opt_state"], state["trainable_variables"]
        )
        # Apply the watchdog recovery multiplier on top of the optimizer
        # schedule. lr_recovery_multiplier is 1.0 in normal training and gets
        # halved by the loop on each hard-spike recovery.
        mult = state["lr_recovery_multiplier"]
        updates = jax.tree.map(lambda u: u * mult, updates)
        candidate_trainable = optax.apply_updates(
            state["trainable_variables"], updates
        )

        # Atomic select: every leaf in (trainable, non_trainable, opt_state)
        # takes the candidate value when step_ok, else reverts to the old
        # value. `jnp.where` is leafwise; the broadcasting is fine because
        # step_ok is a scalar boolean.
        def _pick(new_leaf, old_leaf):
            return jnp.where(step_ok, new_leaf, old_leaf)

        new_trainable = jax.tree.map(_pick, candidate_trainable, state["trainable_variables"])
        new_non_trainable_final = jax.tree.map(_pick, new_non_trainable, state["non_trainable_variables"])
        new_opt_state = jax.tree.map(_pick, candidate_opt_state, state["opt_state"])

        new_state = {
            "trainable_variables": new_trainable,
            "non_trainable_variables": new_non_trainable_final,
            "opt_state": new_opt_state,
            "step": state["step"] + 1,  # data step advances on every call
            "lr_recovery_multiplier": state["lr_recovery_multiplier"],
        }
        # Expose `nan_skipped` (0.0/1.0) in metrics so the loop can count +
        # log how many bad batches we silently reverted. The reported `loss`
        # is still the raw NaN (informational); the loop should NOT feed NaN
        # losses into the watchdog (it would trigger a phantom spike).
        metrics_out = {
            "loss": loss,
            "nan_skipped": jnp.where(step_ok, jnp.float32(0.0), jnp.float32(1.0)),
            **metrics,
        }
        return new_state, metrics_out

    return train_step
