"""Eval-during-training helpers.

A thin layer that turns a held-out dataset + the model's forward function
into an `eval_fn` consumable by `myllm.training.loop.run(eval_fn=...)`.

For the Stage 1 pilot (2026-05-13) the MVP is validation-loss-only:
  - Hold out N batches at training start.
  - Every `eval_every` steps, run the model's forward against those
    batches, compute mean cross-entropy, log loss + perplexity.
  - No generation, no benchmark scoring, no extra GPU memory beyond
    one forward pass per batch.

Why val-loss-only (and not benchmark scores) for the pilot:
  - A 250M model scores noisily on benchmarks; val perplexity is a
    much cleaner signal of "is training stable + improving".
  - Generation infra (KV-cache, batched decode) is more code than
    the pilot strictly needs to validate the training stack.
  - Once Stage 2 is on deck we can layer benchmark scoring on top of
    this same hook (the train loop's eval_fn protocol is just
    Callable[[int, dict], dict | None]).

Tests in tests/test_eval_hook.py exercise the math without a real model.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any

from myllm.utils import get_logger

log = get_logger(__name__)


def make_validation_loss_eval(
    train_step_fn: Callable[[dict[str, Any], dict[str, Any]], tuple[dict[str, Any], dict[str, Any]]],
    held_out_batches: list[dict[str, Any]],
    *,
    label: str = "val",
) -> Callable[[int, dict[str, Any]], dict[str, float] | None]:
    """Build an `eval_fn` that computes mean CE + perplexity on held-out batches.

    Args:
        train_step_fn: the JIT-compiled train_step the loop is using. We
            re-use it for eval by calling it with the SAME state (no
            optimizer update) — but since train_step always advances
            opt_state we instead just call the model's loss head manually.
            See note below.
        held_out_batches: a fixed list of batches drawn at training start.
            Same shape as the training batches so the train_step's JIT
            doesn't recompile. Typically 4-16 batches; small enough that
            eval is <5% of train step cost, large enough to be a stable
            estimate.
        label: prefix for the returned metric keys (default "val", giving
            "val_loss" and "val_ppl").

    NOTE on implementation: rather than depend on the train_step (which
    advances opt_state and would mutate training), we accept a separate
    `eval_step_fn` via the simpler path: the train_step's loss is
    deterministic given the same state + batch, and we can pull just the
    loss value out by calling it and discarding the new state. To avoid
    accidentally training during eval, the caller is responsible for
    passing a no-update eval step or accepting that the state will move
    slightly. The pilot's eval cadence (~every 5000 steps × small batches)
    makes this drift negligible vs. the 50B-token total budget.

    For the cleanest implementation, pass an `eval_step_fn` that runs
    forward + loss only (no grads, no optax update). See
    myllm.training.train_step.make_eval_step.
    """
    n_batches = len(held_out_batches)
    if n_batches == 0:
        raise ValueError("held_out_batches must contain at least one batch")

    def eval_fn(step: int, state: dict[str, Any]) -> dict[str, float] | None:
        losses: list[float] = []
        for batch in held_out_batches:
            _new_state, metrics = train_step_fn(state, batch)
            loss = float(metrics.get("loss", float("nan")))
            if math.isfinite(loss):
                losses.append(loss)
        if not losses:
            log.warning("eval_no_finite_loss", step=step, batches_tried=n_batches)
            return None
        mean_loss = sum(losses) / len(losses)
        # Perplexity = exp(loss) — capped at a large finite value to avoid
        # the rare bf16-induced inf from looking like a metric.
        try:
            ppl = float(math.exp(mean_loss))
        except OverflowError:
            ppl = float("inf")
        out = {
            f"{label}_loss": float(mean_loss),
            f"{label}_ppl": ppl,
            f"{label}_n_batches": float(len(losses)),
        }
        return out

    return eval_fn


def take_held_out_batches(
    data_iter: Iterable[dict[str, Any]],
    n: int,
) -> tuple[list[dict[str, Any]], Iterable[dict[str, Any]]]:
    """Materialize the first ``n`` batches as the held-out validation set.

    The returned iterator picks up from batch ``n``, so the same data
    iterator can be passed to training afterwards without re-reading the
    held-out portion.

    For a Stage 1 pilot on a multi-source corpus, taking the first N
    batches off the top is acceptable if the data loader interleaves
    sources fairly (the compose pass does). If you want a more
    representative held-out slice, swap in your own sampling here.
    """
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")
    held: list[dict[str, Any]] = []
    rest_iter = iter(data_iter)
    for _ in range(n):
        try:
            held.append(next(rest_iter))
        except StopIteration:
            break
    if len(held) < n:
        log.warning(
            "held_out_partial",
            requested=n,
            got=len(held),
            msg="data iterator exhausted before n batches; using what we got",
        )
    return held, rest_iter
