"""MuonClip / QK-clip — Muon's stability safety net at long horizon.

Per Kimi K2 paper (arXiv 2507.20534) — the only public >=1T-param Muon
training to date — Muon was paired with a post-update weight rescale of
W_q and W_k whenever the observed maximum attention logit (qk-dot) on a
batch exceeded a threshold ``t``. Kimi K2 used ``t = 100``. They report
zero training spikes across 15.5T tokens.

Algorithm (per Kimi K2 + the Fireworks AI deep-dive at
https://fireworks.ai/blog/muonclip):

  for each attention layer:
      max_score = max over (B, H, S, S) of q @ k.T * scale
      if max_score > t:
          eta = t / max_score
          W_q *= eta ** alpha       # alpha defaults to 0.5
          W_k *= eta ** (1 - alpha)

This is a per-step WEIGHT rescale, NOT a gradient clip and NOT a logit
clip — the attention logits inside the network are not altered. The
rescale is applied AFTER the optimizer step and BEFORE the next forward.

This implementation is JAX-side, pure-functional, JIT-friendly. The
caller (``train_step``) is responsible for passing in:
  - the current attention-layer Q/K weights
  - the observed max attention logit from the just-finished forward pass
  - the configured threshold ``t``

State-of-the-art caveats (verified 2026-05-18):
  - MuonClip is described textually in the Kimi K2 paper; no canonical
    JAX reference impl exists in optax. Open-source reference:
    https://github.com/AkulDatta/muonclip (JAX + PyTorch, untested at
    1B+).
  - Threshold ``t = 100`` is Kimi K2's production value; we treat it as
    the default but it is empirically tunable per architecture.
  - alpha defaults to 0.5 (symmetric rescale between Q and K).
"""
from __future__ import annotations

from typing import Any


def apply_qk_clip(
    wq: Any,
    wk: Any,
    max_qk_score: Any,
    *,
    threshold: float = 100.0,
    alpha: float = 0.5,
) -> tuple[Any, Any]:
    """Rescale ``W_q``, ``W_k`` in-place-of (functional) when
    ``max_qk_score > threshold``.

    Args:
        wq: ``[H, H]`` query projection weight (any dtype).
        wk: ``[H, H/n_rep]`` key projection weight.
        max_qk_score: scalar — max over (B, H_heads, S, S) of the QK
            product (BEFORE softmax, AFTER scale=1/sqrt(d_head)).
        threshold: trigger value. Kimi K2 uses 100.0.
        alpha: rescale split. 0.5 = symmetric (W_q and W_k both
            scaled by sqrt(eta)). Kimi K2 paper uses 0.5.

    Returns:
        ``(new_wq, new_wk)``. When ``max_qk_score <= threshold`` the
        original tensors pass through unchanged (jnp.where, JIT-safe).

    Mathematical guarantee: post-clip the max attention score is at
    most ``threshold`` (assuming ``q · k`` was the limiting case;
    masked positions are unaffected).
    """
    from keras import ops

    # Compute the rescale factor in a JIT-safe way: avoid Python-level
    # branching on the tracer. Use ops.where to no-op when the score is
    # already under the threshold.
    eta = ops.cast(threshold, max_qk_score.dtype) / max_qk_score
    eta = ops.where(
        max_qk_score > ops.cast(threshold, max_qk_score.dtype),
        eta,
        ops.cast(1.0, eta.dtype),
    )
    # Cast to weight dtype for the rescale.
    eta_q = ops.cast(eta ** alpha, wq.dtype)
    eta_k = ops.cast(eta ** (1.0 - alpha), wk.dtype)
    return wq * eta_q, wk * eta_k


def collect_max_qk_scores(
    diagnostics: dict[str, Any] | None,
) -> Any | None:
    """Pull the per-layer max QK scores out of a diagnostics dict.

    The model forward must populate ``diagnostics["attn_max_qk_per_layer"]``
    as a ``[L]`` array (one max per attention layer) if MuonClip is to be
    applied. When this hook isn't wired (initial Stage 2 smoke), this
    returns None and MuonClip is a no-op for the step.

    Wiring is intentionally minimal in this first version:
      - the model already exposes scores in the manual attention path
        when ``attn_logit_softcap`` is set; collecting the max is cheap
        (one ``ops.max(scores)`` per layer)
      - the cuDNN-Flash fast path does NOT expose intermediate scores,
        so MuonClip + Flash are incompatible (same constraint Gemma 2's
        softcap had — which is why our config defaults softcap off in
        the Gemma-3-era reconciliation)

    Returns:
        Per-layer max QK score array, or None if diagnostics aren't
        present.
    """
    if diagnostics is None:
        return None
    return diagnostics.get("attn_max_qk_per_layer")
