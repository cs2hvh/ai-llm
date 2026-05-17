"""Forward-only eval step factory (FSDP-safe, no donation, no grads).

The training-time eval hook (`myllm.training.eval_hook`) re-uses the
JIT-compiled `train_step_fn` for evaluation. That works for the
non-FSDP case but breaks under FSDP because `train_step` declares
`donate_argnums=(0,)`, which destroys the input state buffer in-place.
Reusing it for eval would corrupt training state on the next training
call.

This module provides a separate `make_eval_step(...)` that:
  - Runs ONE forward pass through the model + the CE loss head.
  - Does NOT take grads, does NOT call the optimizer, does NOT mutate
    state.
  - Does NOT use ``donate_argnums``, so it's safe to call between
    training steps.
  - When ``state_shardings`` is provided, declares ``in_shardings`` so
    the compiled JIT matches the live training topology.
  - Optionally returns ``nll_per_token: [B, S]`` for per-source bucketing
    (consumed by Phase 1.2 P0-1 per-source val loss).

Distillation is intentionally not on the eval path: the pilot's val
metric is "how well does the model predict held-out tokens" — CE / PPL
only. Teacher-mixed loss is a training-time objective.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from myllm.training.loss import (
    chunked_cross_entropy_with_z_loss,
    cross_entropy_with_z_loss,
)


def make_eval_step(
    model: Any,
    *,
    z_loss_coef: float = 1.0e-4,
    ignore_index: int | None = None,
    use_chunked_ce: bool = False,
    chunked_ce_num_chunks: int = 8,
    final_logit_softcap: float | None = None,
    return_per_token_nll: bool = False,
    state_shardings: Any = None,
    batch_sharding: Any = None,
) -> Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]:
    """Return a JIT-compiled ``(state, batch) -> metrics`` function.

    The compiled function:
      - Does NOT mutate ``state``. No optimizer update; no donation.
      - Returns ``metrics`` only (a dict of scalar JAX arrays).
      - Compiles under the same sharding contract as ``train_step`` when
        ``state_shardings`` is supplied — same in_shardings on the state
        pytree and batch — so live training state can be passed in
        without re-placing.

    Args:
        model: Keras model. Must support ``model.stateless_call(...,
            return_loss_inputs=use_chunked_ce, segment_ids=...)``.
        z_loss_coef: matches train_step's z-loss weight so the reported
            ``loss = ce + z_loss_coef * z_loss`` is comparable.
        ignore_index: same as train_step; masks pad-token positions.
        use_chunked_ce: when True, uses the streamed-vocab path
            (no [B,S,V] materialisation). Pair with the same
            ``chunked_ce_num_chunks`` train_step uses.
        return_per_token_nll: when True, metrics include
            ``nll_per_token: [B, S]`` and ``weight_per_token: [B, S]``.
            Used by per-source val loss to bucket tokens by their
            source_id.
        state_shardings: same as train_step's state_shardings argument.
            Pass the live training shardings so the JIT doesn't have to
            re-trace.
        batch_sharding: same as train_step's batch_sharding argument.

    Returns:
        Callable ``eval_step(state, batch) -> metrics``.
    """
    try:
        import jax
    except ImportError as e:
        raise ImportError("jax required for eval_step") from e

    def _forward(
        trainable: Any,
        non_trainable: Any,
        batch: dict[str, Any],
    ) -> dict[str, Any]:
        segment_ids = batch.get("segment_ids")
        loss_mask = batch.get("loss_mask")
        call_kwargs = {}
        if segment_ids is not None:
            call_kwargs["segment_ids"] = segment_ids

        if use_chunked_ce:
            # The chunked-CE forward returns (hidden, lm_head_w, output_mult)
            # via return_loss_inputs=True. No teacher path on eval.
            (hidden, lm_head_w, output_mult), _new_non_trainable = (
                model.stateless_call(
                    trainable, non_trainable, batch["input_ids"],
                    return_loss_inputs=True, **call_kwargs,
                )
            )
            loss, ce_metrics = chunked_cross_entropy_with_z_loss(
                hidden, lm_head_w, batch["labels"],
                num_chunks=chunked_ce_num_chunks,
                output_mult=output_mult,
                ignore_index=ignore_index,
                z_loss_coef=z_loss_coef,
                loss_mask=loss_mask,
                final_logit_softcap=final_logit_softcap,
                return_per_token=return_per_token_nll,
            )
            return {"loss": loss, **ce_metrics}

        # Full-logit forward path (matches train_step's non-chunked branch).
        logits, _new_non_trainable = model.stateless_call(
            trainable, non_trainable, batch["input_ids"], **call_kwargs
        )
        loss, ce_metrics = cross_entropy_with_z_loss(
            logits, batch["labels"],
            ignore_index=ignore_index,
            z_loss_coef=z_loss_coef,
            loss_mask=loss_mask,
            return_per_token=return_per_token_nll,
        )
        return {"loss": loss, **ce_metrics}

    def _eval_step_body(
        state: dict[str, Any], batch: dict[str, Any]
    ) -> dict[str, Any]:
        return _forward(
            state["trainable_variables"],
            state["non_trainable_variables"],
            batch,
        )

    # JIT compile. Two flavors mirroring train_step's:
    #   - No FSDP (state_shardings=None): plain @jax.jit, no donation.
    #   - FSDP: declare in_shardings on (state, batch). NO donate_argnums
    #     — the whole point is to leave state untouched.
    if state_shardings is None:
        eval_step = jax.jit(_eval_step_body)
    else:
        eval_step = jax.jit(
            _eval_step_body,
            in_shardings=(state_shardings, batch_sharding),
        )

    return eval_step
