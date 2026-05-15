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
    ``make_validation_loss_eval_from_eval_step`` below — that is the
    FSDP-safe path (Phase 1.5, 2026-05-15).
    """
    n_batches = len(held_out_batches)
    if n_batches == 0:
        raise ValueError("held_out_batches must contain at least one batch")

    def eval_fn(step: int, state: dict[str, Any]) -> dict[str, float] | None:
        # 2026-05-14: same int32-overflow fix as the train loop (commit
        # 9f442f7). train_step_fn JITs over state leaves and JAX
        # defaults Python ints to int32, which overflows for
        # data_position > 2^31 (~65K steps at mb=4, seq=8192). Eval
        # was silently failing on every call post-step-65500 in the
        # 2026-05-13 pilot. Shallow-copy state without data_position;
        # the original in the loop is untouched.
        eval_state = {k: v for k, v in state.items() if k != "data_position"}
        losses: list[float] = []
        for batch in held_out_batches:
            _new_state, metrics = train_step_fn(eval_state, batch)
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


def make_validation_loss_eval_from_eval_step(
    eval_step_fn: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]],
    held_out_batches: list[dict[str, Any]],
    *,
    label: str = "val",
) -> Callable[[int, dict[str, Any]], dict[str, float] | None]:
    """Build an `eval_fn` from a forward-only ``eval_step_fn``.

    Use this with ``myllm.training.eval_step.make_eval_step(...)``. The
    difference vs. ``make_validation_loss_eval`` is that ``eval_step_fn``
    returns a metrics dict (no new state) and does NOT donate state
    buffers, so it is safe to call between FSDP-donated train steps.

    Phase 1.5 (2026-05-15): introduced so the loop can run eval under
    ``--fsdp`` without train_step's donate_argnums clobbering state.

    The data_position pop guard is no longer needed: the eval_step's
    JIT in_shardings include data_position when the live state has it,
    or it's absent entirely. Either way no int32 cast happens.
    """
    n_batches = len(held_out_batches)
    if n_batches == 0:
        raise ValueError("held_out_batches must contain at least one batch")

    def eval_fn(step: int, state: dict[str, Any]) -> dict[str, float] | None:
        losses: list[float] = []
        for batch in held_out_batches:
            metrics = eval_step_fn(state, batch)
            loss = float(metrics.get("loss", float("nan")))
            if math.isfinite(loss):
                losses.append(loss)
        if not losses:
            log.warning("eval_no_finite_loss", step=step, batches_tried=n_batches)
            return None
        mean_loss = sum(losses) / len(losses)
        try:
            ppl = float(math.exp(mean_loss))
        except OverflowError:
            ppl = float("inf")
        return {
            f"{label}_loss": float(mean_loss),
            f"{label}_ppl": ppl,
            f"{label}_n_batches": float(len(losses)),
        }

    return eval_fn


def build_per_source_held_out(
    reader: Any,
    *,
    n_sequences: int,
    micro_batch_size: int,
    source_vocab: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], list[Any], dict[str, int]]:
    """Build held-out batches annotated with per-token source-ids.

    For Phase 1.2 (P0-1 per-source val loss). Pairs with
    ``make_per_source_validation_loss_eval_from_eval_step`` below.

    Args:
        reader: ``PackedCorpusReader``. We read the first ``n_sequences``
            sequences from the reader and bucket their tokens by
            ``DocSpan.source_id``. Sequences are taken off the head, not
            randomly sampled — the same held-out slice should be skipped
            during training (see ``run_pretrain.py``'s held-out cursor).
        n_sequences: how many packed sequences to use for the held-out
            set. They get bucketed into ``ceil(n_sequences /
            micro_batch_size)`` batches.
        micro_batch_size: batch size. Must match the training batch size
            so the JIT'd ``eval_step`` doesn't recompile.
        source_vocab: optional pre-built ``source_name -> int_id`` map.
            When ``None`` (the typical case), the vocab is derived from
            the actual sources seen in the held-out slice. The returned
            vocab is the one actually used; pass it back to the eval-fn
            factory.

    Returns:
        Tuple ``(batches, source_id_arrays, vocab)`` where:
          - ``batches``: list of batch dicts ``{"input_ids", "labels",
            "segment_ids", "loss_mask"}`` each of shape ``[B, S]`` —
            same shape the training batch iterator produces, so the
            ``eval_step`` JIT specialisation matches.
          - ``source_id_arrays``: parallel list of ``[B, S]`` int32
            arrays. ``source_id_arrays[i][b, s]`` is the
            ``source_vocab`` int for the LABEL token at position
            ``(b, s)`` in ``batches[i]``. ``-1`` for padding / boundary
            positions (matches segment_ids -1 sentinel).
          - ``vocab``: ``source_name -> int_id`` dict actually used.
            Sorted by source name for deterministic ordering.

    The label-position-vs-input-position distinction matters:
        input_ids[i] is tokens[i]; labels[i] is tokens[i+1]. The loss
        the model pays at position i is for predicting tokens[i+1].
        So the SOURCE we attribute the loss to is whatever produced
        tokens[i+1] — i.e. the source labels are sliced from the
        full-sequence source array as ``source[1:]``.
    """
    import numpy as np

    # Pass 1: discover all sources in the held-out slice. We need the
    # vocab to be complete before we can dictionary-encode any sequence.
    sequence_ids = list(range(min(n_sequences, int(reader.total_sequences))))
    if source_vocab is None:
        seen_sources: set[str] = set()
        for sid in sequence_ids:
            spans = reader.get_provenance(sid)
            for span in spans:
                seen_sources.add(span.source_id)
        source_vocab = {name: i for i, name in enumerate(sorted(seen_sources))}

    # Pass 2: pack sequences into batches. We mirror the training
    # batch_pairs layout: micro_batch_size sequences -> one batch of
    # shape [B, S] where S = sequence_length - 1 (next-token shift).
    batches: list[dict[str, Any]] = []
    source_id_arrays: list[Any] = []

    inputs_buf: list[list[int]] = []
    labels_buf: list[list[int]] = []
    segs_buf: list[list[int]] = []
    masks_buf: list[list[int]] = []
    source_buf: list[list[int]] = []

    for sid in sequence_ids:
        tokens = reader.get_sequence(sid)
        if tokens.shape[0] < 2:
            continue
        seg_ids = reader.get_segment_ids(sid)
        per_token_source = reader.get_per_token_source_ids(sid, source_vocab)

        token_list = [int(t) for t in tokens]
        seg_list = [int(s) for s in seg_ids]
        src_list = [int(s) for s in per_token_source]

        input_ids = token_list[:-1]
        labels = token_list[1:]
        input_segments = seg_list[:-1]
        label_segments = seg_list[1:]
        loss_mask = [
            1 if (a == b and a != -1) else 0
            for a, b in zip(input_segments, label_segments, strict=False)
        ]
        # The label at position i comes from tokens[i+1], so its source
        # is the source assigned to position i+1 in the original packed
        # sequence. Slicing [1:] gives that.
        label_sources = src_list[1:]

        inputs_buf.append(input_ids)
        labels_buf.append(labels)
        segs_buf.append(input_segments)
        masks_buf.append(loss_mask)
        source_buf.append(label_sources)

        if len(inputs_buf) == micro_batch_size:
            batches.append({
                "input_ids": np.asarray(inputs_buf, dtype=np.int32),
                "labels": np.asarray(labels_buf, dtype=np.int32),
                "segment_ids": np.asarray(segs_buf, dtype=np.int32),
                "loss_mask": np.asarray(masks_buf, dtype=np.int32),
            })
            source_id_arrays.append(
                np.asarray(source_buf, dtype=np.int32)
            )
            inputs_buf.clear()
            labels_buf.clear()
            segs_buf.clear()
            masks_buf.clear()
            source_buf.clear()

    # Discard any tail that doesn't fill a full batch — eval_step's JIT
    # specialises on [B, S]; partial batches would re-trigger compile.

    return batches, source_id_arrays, dict(source_vocab)


def make_per_source_validation_loss_eval_from_eval_step(
    eval_step_fn: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]],
    held_out_batches: list[dict[str, Any]],
    source_id_arrays: list[Any],
    source_vocab: dict[str, int],
    *,
    label: str = "val",
) -> Callable[[int, dict[str, Any]], dict[str, float] | None]:
    """Eval-fn that returns aggregate + per-source val_loss + val_ppl.

    Phase 1.2 (P0-1 per-source val loss, 2026-05-15).

    Wraps a forward-only ``eval_step_fn`` built with
    ``make_eval_step(return_per_token_nll=True, ...)``. The eval_step
    surfaces ``nll_per_token: [B, S]`` and ``weight_per_token: [B, S]``;
    this function buckets them by ``source_id_arrays`` and reports:

      - ``<label>_loss``      aggregate CE across all non-masked tokens
      - ``<label>_ppl``       exp of the aggregate CE
      - ``<label>_n_tokens``  number of tokens counted in the aggregate
      - ``<label>_loss/<src>`` per-source CE
      - ``<label>_ppl/<src>``  per-source perplexity
      - ``<label>_n_tokens/<src>``  per-source token count

    Sources with zero non-masked tokens in the held-out are skipped.

    Note vs. the legacy ``make_validation_loss_eval``: the legacy path
    reported ``ce + z_loss_coef * z_loss`` (the training objective).
    This path reports pure CE (the per-token NLL), so the val_ppl here
    is the conventional perplexity (= exp(ce)) rather than the
    training-objective ppl. The two differ by ``z_loss_coef * z_loss``
    which is ~1e-4 — negligible for reporting.
    """
    if len(held_out_batches) != len(source_id_arrays):
        raise ValueError(
            f"batches ({len(held_out_batches)}) and source_id_arrays "
            f"({len(source_id_arrays)}) must have equal length"
        )
    if not held_out_batches:
        raise ValueError("need at least one held-out batch")

    import numpy as np
    inv_vocab = {v: k for k, v in source_vocab.items()}

    def eval_fn(step: int, state: dict[str, Any]) -> dict[str, float] | None:
        # Per-source accumulators.
        src_nll_sum = {name: 0.0 for name in source_vocab}
        src_weight_sum = {name: 0.0 for name in source_vocab}
        total_nll = 0.0
        total_weight = 0.0
        any_finite = False

        for batch, src_ids in zip(
            held_out_batches, source_id_arrays, strict=False,
        ):
            metrics = eval_step_fn(state, batch)
            nll = np.asarray(metrics["nll_per_token"], dtype=np.float64)
            w = np.asarray(metrics["weight_per_token"], dtype=np.float64)
            if not np.all(np.isfinite(nll)):
                # NaN-skip pattern: a non-finite NLL means a bad batch;
                # treat it as "no information" rather than poisoning the
                # aggregate. (Single batches don't get NaN-reverted under
                # eval — the train loop's atomic revert only applies to
                # training steps.)
                log.warning(
                    "eval_per_source_batch_skipped_nan",
                    step=step,
                    batch_max_nll=float(np.nanmax(nll)),
                )
                continue
            any_finite = True

            batch_w = nll * w
            total_nll += float(batch_w.sum())
            total_weight += float(w.sum())

            for src_int, src_name in inv_vocab.items():
                mask = (src_ids == src_int).astype(np.float64) * w
                src_nll_sum[src_name] += float((nll * mask).sum())
                src_weight_sum[src_name] += float(mask.sum())

        if not any_finite or total_weight <= 0.0:
            log.warning(
                "eval_per_source_no_data",
                step=step,
                batches_tried=len(held_out_batches),
            )
            return None

        out: dict[str, float] = {}
        agg_loss = total_nll / total_weight
        out[f"{label}_loss"] = float(agg_loss)
        out[f"{label}_ppl"] = (
            float(math.exp(min(agg_loss, 50.0))) if math.isfinite(agg_loss) else float("inf")
        )
        out[f"{label}_n_tokens"] = float(total_weight)
        out[f"{label}_n_batches"] = float(len(held_out_batches))

        for name in sorted(source_vocab):
            sw = src_weight_sum[name]
            if sw <= 0.0:
                continue
            src_loss = src_nll_sum[name] / sw
            out[f"{label}_loss/{name}"] = float(src_loss)
            out[f"{label}_ppl/{name}"] = (
                float(math.exp(min(src_loss, 50.0)))
                if math.isfinite(src_loss) else float("inf")
            )
            out[f"{label}_n_tokens/{name}"] = float(sw)
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
