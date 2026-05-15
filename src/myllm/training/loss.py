"""Loss functions for language-model training.

Cross-entropy is the primary loss. We add a small ``z-loss`` term that
penalises the log-partition function. This stabilises late-training dynamics
by preventing logit norms from drifting (PaLM 2022, Chowdhery et al.).

R0 (2026-05-11): added top-K logit distillation loss for the decay-phase
multi-teacher recipe locked in
``docs/governance/teacher_distillation_strategy.md``. The teacher logits are
pre-cached offline (top-K per token per teacher; production K=64 per the
locked teacher plan). During training we mix cross-entropy and per-teacher
KL divergence at the caller-specified ``alpha`` ratio.

2026-05-12 (reviewer pushback): replaced ``one_hot(labels) * log_softmax``
in CE with ``take_along_axis`` (gather). The one-hot materialised a
``[B, S, V]`` float tensor — 8.6 GB at B=8 S=4097 V=131072 bf16 — for no
information gain over a gather. Equivalence locked by ``tests/test_loss.py``.
Full-logit memory is still ``[B, S, V]`` (chunked-LM-head CE is a separate
follow-up; see ``docs/governance/model_card_v1.md`` perf section).

All ops go through ``keras.ops`` so the same code runs on JAX or TF.
"""
from __future__ import annotations

from typing import Any


def cross_entropy_with_z_loss(
    logits: Any,
    labels: Any,
    *,
    ignore_index: int | None = None,
    z_loss_coef: float = 1.0e-4,
    loss_mask: Any | None = None,
    return_per_token: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Token-level cross-entropy + z-loss.

    Args:
        logits: ``[batch, seq, vocab]`` un-normalised logits.
        labels: ``[batch, seq]`` int targets.
        ignore_index: if set, positions equal to this value contribute 0 loss
            and 0 weight (typically the pad-token id).
        z_loss_coef: weight on the z-loss term ``mean(logsumexp(logits)^2)``.
        loss_mask: optional ``[batch, seq]`` int/bool array. Positions with 0
            contribute 0 loss and 0 weight. Used for intra-document boundary
            masking (R2): we don't want to penalize the model for failing to
            predict doc B's first token from doc A's content. AND-combined
            with the ignore_index check if both are set.
        return_per_token: when True, the metrics dict additionally contains
            ``nll_per_token: [B, S]`` (raw NLL, BEFORE weight masking) and
            ``weight_per_token: [B, S]`` (the combined ignore_index+loss_mask
            weight). The caller can do its own grouped reduction (e.g.,
            per-source bucketed mean for P0-1 per-source val loss). When
            no weight applies, ``weight_per_token`` is all-ones.

    Returns:
        ``(loss, metrics_dict)``. The metrics dict contains ``ce`` and
        ``z_loss`` for separate logging.
    """
    from keras import ops

    log_z = ops.logsumexp(logits, axis=-1)  # [b, s]
    log_softmax = logits - log_z[..., None]  # [b, s, v]
    # Gather log-prob at target positions. Equivalent to
    #   nll = -sum(one_hot(labels) * log_softmax, axis=-1)
    # but does not materialise the [B, S, V] one-hot tensor (8.6 GB at
    # B=8 S=4097 V=131072 bf16). The full [B,S,V] log_softmax tensor is
    # still allocated; collapsing that requires chunked-LM-head CE (TBD).
    labels_int = ops.cast(labels, "int32")
    gathered = ops.take_along_axis(log_softmax, labels_int[..., None], axis=-1)
    nll = -ops.squeeze(gathered, axis=-1)  # [b, s]

    # Combine ignore_index + loss_mask into a single weight tensor.
    weight = None
    if ignore_index is not None:
        weight = ops.cast(ops.not_equal(labels, ignore_index), nll.dtype)
    if loss_mask is not None:
        mask = ops.cast(loss_mask, nll.dtype)
        weight = mask if weight is None else (weight * mask)

    if weight is not None:
        nll_sum = ops.sum(nll * weight)
        denom = ops.maximum(ops.sum(weight), ops.cast(1.0, weight.dtype))
        ce = nll_sum / denom
        z_loss = ops.sum((log_z * log_z) * weight) / denom
    else:
        ce = ops.mean(nll)
        z_loss = ops.mean(log_z * log_z)

    total = ce + z_loss_coef * z_loss
    metrics: dict[str, Any] = {"ce": ce, "z_loss": z_loss}
    if return_per_token:
        metrics["nll_per_token"] = nll
        metrics["weight_per_token"] = (
            weight if weight is not None else ops.ones_like(nll)
        )
    return total, metrics


def chunked_cross_entropy_with_z_loss(
    hidden_states: Any,
    lm_head_weight: Any,
    labels: Any,
    *,
    num_chunks: int = 8,
    output_mult: float = 1.0,
    ignore_index: int | None = None,
    z_loss_coef: float = 1.0e-4,
    loss_mask: Any | None = None,
    return_per_token: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Cross-entropy + z-loss that never materialises the full [B, S, V] logit tensor.

    Streams the vocabulary in ``num_chunks`` slices, computes per-chunk
    logits ``[B, S, V/num_chunks]``, and accumulates the global log-sum-exp
    online (Numerically stable trick: ``log Σ exp(x) = m + log Σ exp(x - m)``
    extended across chunks by carrying ``(running_max, running_sum)``). The
    target-position logit is gathered from whichever chunk contains its
    label index — using ``where(labels in chunk, gather, carry)``.

    Peak transient logit memory drops from ``[B, S, V]`` to ``[B, S,
    V/num_chunks]``. At B=8, S=4097, V=131072 in bf16: full = 8.6 GB,
    chunked with num_chunks=8 = 1.1 GB. (Hidden states + LM-head weight
    are unchanged.)

    Args:
        hidden_states: ``[B, S, H]`` — model output BEFORE the LM head.
            Get this from ``TransformerLM._compute_hidden(ids, ...)``.
        lm_head_weight: ``[V, H]`` — the tied embedding matrix (or the
            transposed untied LM head kernel). Pass
            ``model.lm_head_weight``.
        labels: ``[B, S]`` int targets.
        num_chunks: vocab is split into this many equal chunks.
            ``V % num_chunks == 0`` is required (asserted). Production
            default 8 (V=131072 → chunks of 16384).
        output_mult: muP LM-head output multiplier. Caller passes
            ``model.lm_head_output_mult`` (1.0 when muP off).
        ignore_index, z_loss_coef, loss_mask: as in
            ``cross_entropy_with_z_loss``.

    Returns:
        ``(loss, {"ce": ..., "z_loss": ...})`` — numerically equivalent
        to ``cross_entropy_with_z_loss`` on the full logits within ~1e-4
        in bf16, ~1e-6 in float32 (locked by tests/test_loss.py).
    """
    from keras import ops

    # Static shapes — JIT requires this so per-chunk slicing is statically
    # known. lm_head_weight is a weight, so its shape is always static.
    weight_shape = lm_head_weight.shape
    if len(weight_shape) != 2:
        raise ValueError(
            f"lm_head_weight must be [V, H]; got shape {weight_shape}"
        )
    vocab_size = int(weight_shape[0])
    if vocab_size % num_chunks != 0:
        raise ValueError(
            f"vocab_size ({vocab_size}) must be divisible by num_chunks "
            f"({num_chunks}); pad the vocab or pick a different chunk count."
        )
    chunk_size = vocab_size // num_chunks

    labels_int = ops.cast(labels, "int32")

    # Sentinel: NEG_INF for the running max so the first chunk's max wins.
    # Don't use float("-inf"): mixed-precision ops can NaN on -inf - -inf.
    neg_inf = ops.cast(-1.0e30, hidden_states.dtype)
    running_max = ops.full_like(labels_int, neg_inf, dtype=hidden_states.dtype)
    running_sum = ops.zeros_like(running_max)
    label_logits = ops.zeros_like(running_max)

    output_mult_t = ops.cast(output_mult, hidden_states.dtype)

    for c in range(num_chunks):
        lo = c * chunk_size
        # Slice on a Python int — static under JIT.
        chunk_embed = lm_head_weight[lo : lo + chunk_size]  # [chunk_size, H]
        # Per-chunk logits: hidden @ chunk_embed.T  ->  [B, S, chunk_size]
        chunk_logits = ops.matmul(hidden_states, ops.transpose(chunk_embed))
        chunk_logits = chunk_logits * output_mult_t

        # Online logsumexp update (stable across chunks).
        chunk_max = ops.max(chunk_logits, axis=-1)  # [B, S]
        new_max = ops.maximum(running_max, chunk_max)
        running_sum = running_sum * ops.exp(running_max - new_max) + ops.sum(
            ops.exp(chunk_logits - new_max[..., None]), axis=-1
        )
        running_max = new_max

        # Gather label logit IFF labels fall in this chunk's range.
        local_idx = labels_int - lo
        in_chunk = (labels_int >= lo) & (labels_int < lo + chunk_size)
        # Clip for OOB safety on positions outside the chunk (their value
        # is discarded by the `where` below; the clip just keeps the gather
        # from being undefined behavior).
        local_idx_safe = ops.clip(local_idx, 0, chunk_size - 1)
        gathered = ops.take_along_axis(
            chunk_logits, local_idx_safe[..., None], axis=-1
        )
        gathered = ops.squeeze(gathered, axis=-1)
        label_logits = ops.where(in_chunk, gathered, label_logits)

    log_z = running_max + ops.log(running_sum)  # [B, S]
    nll = -(label_logits - log_z)  # [B, S]

    # Same reduction as cross_entropy_with_z_loss.
    weight = None
    if ignore_index is not None:
        weight = ops.cast(ops.not_equal(labels_int, ignore_index), nll.dtype)
    if loss_mask is not None:
        mask = ops.cast(loss_mask, nll.dtype)
        weight = mask if weight is None else (weight * mask)

    if weight is not None:
        nll_sum = ops.sum(nll * weight)
        denom = ops.maximum(ops.sum(weight), ops.cast(1.0, weight.dtype))
        ce = nll_sum / denom
        z_loss = ops.sum((log_z * log_z) * weight) / denom
    else:
        ce = ops.mean(nll)
        z_loss = ops.mean(log_z * log_z)

    total = ce + z_loss_coef * z_loss
    metrics: dict[str, Any] = {"ce": ce, "z_loss": z_loss}
    if return_per_token:
        metrics["nll_per_token"] = nll
        metrics["weight_per_token"] = (
            weight if weight is not None else ops.ones_like(nll)
        )
    return total, metrics


def kl_div_topk_loss(
    student_logits: Any,
    teacher_topk_logits: Any,
    teacher_topk_indices: Any,
    *,
    temperature: float = 1.0,
    ignore_index: int | None = None,
    labels: Any | None = None,
    loss_mask: Any | None = None,
) -> Any:
    """KL divergence between student and teacher distributions on the
    teacher's top-K vocab positions.

    The teacher's distribution is given by ``softmax(teacher_topk_logits / T)``
    over K positions specified by ``teacher_topk_indices``. The student's
    distribution at those same K positions is gathered from its full-vocab
    logits and renormalised, then we compute KL(teacher || student).

    This is the standard "restricted top-K" distillation loss (Hinton et
    al. 2015; used in essentially every distilled small-LLM cited in
    ``docs/ai_research_dossier_2026-05-11.md``). The "mass outside top-K"
    in the teacher's full distribution is implicitly dropped — fine when K
    is large enough to capture the bulk. Production K=64 per the locked
    teacher plan; the per-teacher top-K mass audit (TBD) verifies that K=64
    actually captures >=99% mass on code/math distributions before full
    teacher-cache generation.

    Args:
        student_logits:        ``[B, S, V]`` student's full-vocab logits.
        teacher_topk_logits:   ``[B, S, K]`` teacher's top-K logit values
                               (pre-cached offline; same vocab as student).
        teacher_topk_indices:  ``[B, S, K]`` int indices into ``V``.
        temperature:           teacher-side softmax temperature. ``1.0``
                               matches the teacher's natural distribution;
                               ``>1`` softens it (more mass on tail).
        ignore_index, labels:  if both provided, positions where
                               ``labels == ignore_index`` contribute zero
                               loss (typically pad-token id).

    Returns:
        Scalar KL divergence, averaged over (B*S) effective positions.
    """
    from keras import ops

    # Student's logits at the teacher's top-K positions: [B, S, K]
    student_at_topk = ops.take_along_axis(
        student_logits, teacher_topk_indices, axis=-1
    )

    # Teacher distribution over top-K (with temperature).
    teacher_logits_T = teacher_topk_logits / temperature
    teacher_log_z = ops.logsumexp(teacher_logits_T, axis=-1, keepdims=True)
    teacher_log_p = teacher_logits_T - teacher_log_z
    teacher_p = ops.exp(teacher_log_p)

    # Student distribution renormalised over the same K positions.
    student_log_z = ops.logsumexp(student_at_topk, axis=-1, keepdims=True)
    student_log_p = student_at_topk - student_log_z

    # KL(teacher || student) = sum_k p_t * (log p_t - log p_s)
    kl_per_position = ops.sum(
        teacher_p * (teacher_log_p - student_log_p), axis=-1
    )  # [B, S]

    weight = None
    if ignore_index is not None and labels is not None:
        weight = ops.cast(
            ops.not_equal(labels, ignore_index), kl_per_position.dtype
        )
    if loss_mask is not None:
        mask = ops.cast(loss_mask, kl_per_position.dtype)
        weight = mask if weight is None else (weight * mask)
    if weight is not None:
        denom = ops.maximum(ops.sum(weight), ops.cast(1.0, weight.dtype))
        return ops.sum(kl_per_position * weight) / denom
    return ops.mean(kl_per_position)


def multi_teacher_kl_loss(
    student_logits: Any,
    teacher_topk_logits_per_teacher: Any,
    teacher_topk_indices_per_teacher: Any,
    *,
    teacher_weights: tuple[float, ...] | None = None,
    temperature: float = 1.0,
    ignore_index: int | None = None,
    labels: Any | None = None,
    loss_mask: Any | None = None,
) -> Any:
    """Average KL loss across multiple teachers.

    Teachers are stacked along a leading axis. We compute the top-K KL per
    teacher and average (weighted, if ``teacher_weights`` is provided).

    Args:
        student_logits: ``[B, S, V]``.
        teacher_topk_logits_per_teacher:  ``[T, B, S, K]``.
        teacher_topk_indices_per_teacher: ``[T, B, S, K]`` int.
        teacher_weights:   optional ``[T]`` weights (default: uniform).
        temperature, ignore_index, labels: as in ``kl_div_topk_loss``.

    Returns:
        Scalar weighted-average KL.
    """
    from keras import ops

    n_teachers = ops.shape(teacher_topk_logits_per_teacher)[0]

    # Iterate teachers via Python loop — JAX/JIT folds it. The teacher
    # axis is small (1-3) so an explicit loop is clearer than a vmap.
    losses = []
    for t in range(int(n_teachers)):
        losses.append(
            kl_div_topk_loss(
                student_logits,
                teacher_topk_logits_per_teacher[t],
                teacher_topk_indices_per_teacher[t],
                temperature=temperature,
                ignore_index=ignore_index,
                loss_mask=loss_mask,
                labels=labels,
            )
        )
    stacked = ops.stack(losses)  # [T]

    if teacher_weights is None:
        return ops.mean(stacked)
    weights = ops.cast(ops.convert_to_tensor(teacher_weights), stacked.dtype)
    weights = weights / ops.sum(weights)
    return ops.sum(stacked * weights)


def distillation_mixed_loss(
    student_logits: Any,
    labels: Any,
    teacher_topk_logits_per_teacher: Any | None,
    teacher_topk_indices_per_teacher: Any | None,
    *,
    alpha: float = 0.3,
    teacher_weights: tuple[float, ...] | None = None,
    temperature: float = 1.0,
    ignore_index: int | None = None,
    z_loss_coef: float = 1.0e-4,
    loss_mask: Any | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Combined cross-entropy + multi-teacher KL distillation loss.

    Loss = ``α · CE(student, labels) + (1 - α) · mean_t KL(teacher_t || student)``

    When teacher data is absent (None), the function collapses to plain
    cross-entropy + z-loss — i.e. the stable phase of training, before
    decay-phase distillation kicks in.

    Args:
        student_logits:                ``[B, S, V]``.
        labels:                        ``[B, S]`` int.
        teacher_topk_logits_per_teacher,
        teacher_topk_indices_per_teacher:
            ``[T, B, S, K]`` or ``None`` for "no teacher this batch".
        alpha:           CE weight. ``1.0`` → pure CE (no distillation).
                         ``0.3`` is our locked decay-phase value
                         (``docs/teacher_distillation_strategy.md``).
        teacher_weights: per-teacher weights; defaults to uniform.
        temperature:     teacher softmax temperature.
        ignore_index:    pad-token id.
        z_loss_coef:     z-loss weight on the CE term (unchanged from
                         non-distillation path).

    Returns:
        ``(loss, metrics_dict)`` — metrics include ``ce``, ``z_loss``,
        ``kl`` (zero when no teacher), ``alpha``.
    """
    from keras import ops

    ce_total, ce_metrics = cross_entropy_with_z_loss(
        student_logits, labels,
        ignore_index=ignore_index,
        z_loss_coef=z_loss_coef,
        loss_mask=loss_mask,
    )

    if teacher_topk_logits_per_teacher is None or teacher_topk_indices_per_teacher is None:
        # No teachers this batch — return CE alone (with zero KL recorded
        # so metrics stay consistent across phases).
        metrics = dict(ce_metrics)
        metrics["kl"] = ops.cast(0.0, ce_total.dtype)
        metrics["alpha"] = ops.cast(1.0, ce_total.dtype)
        return ce_total, metrics

    kl_total = multi_teacher_kl_loss(
        student_logits,
        teacher_topk_logits_per_teacher,
        teacher_topk_indices_per_teacher,
        teacher_weights=teacher_weights,
        temperature=temperature,
        ignore_index=ignore_index,
        labels=labels,
        loss_mask=loss_mask,
    )

    total = alpha * ce_total + (1.0 - alpha) * kl_total
    metrics = dict(ce_metrics)
    metrics["kl"] = kl_total
    metrics["alpha"] = ops.cast(alpha, ce_total.dtype)
    return total, metrics
