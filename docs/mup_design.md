# muP / muTransfer Design — v1 (2026-05-11)

**Decision owner:** harshit.hv@samatva.com
**Status:** IMPLEMENTATION IN PROGRESS. R1 from `docs/ai_research_dossier_2026-05-11.md`.

This doc captures the design and math for adding **Maximal Update
Parameterization (muP)** to MyLLM. The goal is **zero-shot hyperparameter
transfer** from a tiny 30M proxy → 250M pilot → 1B base, so we don't have to
re-tune LR / init at each scale (which would cost an extra pilot-equivalent
each time and risk LR misfit at base where rollbacks cost $10-40K).

## What muP buys us

Standard parameterization (SP, what we currently use):
- Init std = 0.02 for all weights
- One global LR
- Optimal LR is **width-dependent** — what's stable at 768 hidden may diverge at 2048
- Forces "tune at every scale" — at frontier labs this is the dominant pilot cost

Maximal Update Parameterization (muP):
- Init and per-layer LR scale **systematically** with width
- Optimal LR derived at small scale (~30M) **transfers exactly** to large (~1B+)
- Reduces variance of "what LR should I use at base" from ±2× to ±1.1×
- Already used by: MiniCPM (their "model wind tunnel"), Qwen 2.5 series, Phi-3, multiple Anthropic models

References: Yang et al. arXiv:2203.03466 (Tensor Programs V), EleutherAI blog
"A Practitioner's Guide to muP" (https://blog.eleuther.ai/mutransfer/),
MiniCPM tech report arXiv:2404.06395.

## Our recipe (the practical minimal muP)

We adopt the **EleutherAI "minimal muP"** variant — three additive changes,
all controlled by one config knob `mup.base_width`. When `mup` is unset
the model behaves exactly as before (SP). When `mup.base_width` is set,
the three scaling factors below activate.

Let
```
W_target = config.hidden_dim
W_base   = config.mup.base_width        (e.g., 256 — the width at which we
                                         tune HPs in the wind tunnel)
m        = W_target / W_base            ("width multiplier")
```

For our planned configs:

| Config | hidden_dim | width_mult vs base=256 |
|---|---|---|
| Wind tunnel (30M proxy) | 384 | 1.5 |
| Pilot 250M | 768 | 3.0 |
| Base 1B | 2048 | 8.0 |

### Change 1: Attention output multiplier

In each `GroupedQueryAttention.call()`, after the `wo` projection, multiply
the output by `attn_output_mult = 1 / m`:

```python
out = matmul(attn_out, wo) * attn_output_mult
```

Why: scales the attention block's contribution to the residual stream so
the signal magnitude is invariant to width. In SP this scaling is implicit
in the init (output projections init with smaller std at wider models); in
muP we make it explicit so init can stay constant.

### Change 2: FFN output multiplier

Same idea — in `SwiGLUFFN.call()`, after `w_down`, multiply by
`ffn_output_mult = 1 / m`. Same justification.

### Change 3: LM head output multiplier

In `TransformerLM.call()`, after computing logits, multiply by
`lm_head_output_mult = 1 / m`. Keeps logit magnitudes invariant to width.

### Change 4: Per-parameter LR scaling (deferred to optim.py)

The full muP recipe also rescales the optimizer LR per parameter:

| Parameter group | LR scale |
|---|---|
| Token embedding | `1` (unchanged) |
| LM head (if untied) | `1` (unchanged) |
| Norm scales (RMSNorm `weight`) | `1` (unchanged) |
| Hidden weights (Q, K, V, O, gate, up, down) | `1 / m` (LR shrinks at wider models) |

This change lives in `src/myllm/training/optim.py` (not yet implemented in
this PR). It uses Optax's `multi_transform` to apply different LR multipliers
to different parameter groups.

## Initialization rules (unchanged)

We keep our existing init: `RandomNormal(std=0.02)` for all weights, with
`scaled_init_for_residuals=True` on the base 1B (the `std/sqrt(2*L)` factor
on output projections). Minimal muP **does not require changing init** —
the output-multiplier and per-param-LR changes together provide the same
invariance the variant-init approach gives.

We could later switch to a "muP-pure" parameterization where init also
scales with width, but the minimal variant is cheaper to maintain and
empirically equivalent at our scale band (per EleutherAI ablations).

## The wind-tunnel sweep — how we actually use muP

Once the three multipliers + per-param-LR are in place:

1. Build a **30M proxy model**: `hidden=384, layers=8, ffn=1536, n_heads=6, n_kv=2, head_dim=64, vocab=131072, context=2048`. Param count ~30M.
2. Sweep over **5-10 LR values** (e.g., 5e-4 → 8e-3, log-spaced) × **2 init scales** (0.01, 0.02) × **2 schedule shapes** (constant + cosine warmup).
3. Each sweep config trains for **1 B tokens** on the 30M proxy. Cost: ~$200-300 per cell, ~$2-3K for full sweep.
4. Pick the LR / init that gives the lowest validation loss on a held-out subset of the pretrain mix.
5. **Apply those exact values to pilot 250M (`mup.base_width=256`, `m=3`) and base 1B (`m=8`)** — no further tuning.

The "zero-shot transfer" claim is that the LR which is optimal at 30M is
also optimal at 1B under muP. With SP this is false — base would need its
own sweep, costing 5-10× the proxy sweep at base scale (i.e. ~$10-30K).

## Validation plan

Before trusting muP at the 1B base run, we verify the transfer empirically:

1. **Width-invariance test (cheap, unit test).** Forward a fixed input
   through two models with `width_mult=1` and `width_mult=4` and the same
   muP-scaled config. Check that **layer-wise activation magnitudes are
   within 1.5× of each other** (vs ~width-fold divergence under SP).

2. **Loss-coordinate test (sweep validation, $2-3K).** Train 30M and 100M
   proxies with the same LR (selected from the sweep). Check that both
   land at the same training loss within ±5% at fixed step count. Under
   SP this fails (LR ideal for 30M diverges at 100M). Under muP it should
   pass.

3. **Pilot transfer test (already-planned Phase 2 pilot is the test).**
   Run pilot 250M with `mup.base_width=256, LR=<from sweep>`. Compare to
   a control run with the same LR but no muP scaling. The muP run should
   reach lower / equal loss at fixed step count.

If any of these three checks fail, we revert to SP and accept the LR
misfit risk at base scale.

## File touch list (this PR + follow-ups)

This PR (scaffolding, default-off):
- [x] `docs/mup_design.md` ← this file
- [x] `src/myllm/model/config.py` — add `MupConfig` optional field
- [x] `src/myllm/model/layers.py` — accept output multipliers in `GroupedQueryAttention` and `SwiGLUFFN`; default to 1.0 (no-op)
- [x] `src/myllm/model/transformer.py` — accept LM-head output multiplier; default 1.0
- [x] `tests/test_mup.py` — width-invariance + scale-correctness regression tests

Follow-up PR (active muP):
- [ ] `src/myllm/training/optim.py` — per-parameter LR scaling via Optax `multi_transform`
- [ ] `configs/wind_tunnel.yaml` — 30M proxy config
- [ ] `scripts/wind_tunnel_sweep.py` — execute the LR/init sweep, write results
- [ ] `tests/test_mup_optim.py` — verify LR scaling is applied per param group
- [ ] Update `configs/pilot_250m.yaml` + `configs/base_1b.yaml` with `mup.base_width: 256`

## What "best possible 1B" gets from muP specifically

1. **Lower LR misfit risk at base.** Without muP, the base 1B's LR is a
   semi-educated guess; with muP it's derived from cheap sweep data.
2. **No "tune at every scale" cost.** Saves an extra pilot-equivalent run
   (~$500-1000) that we would otherwise need at base scale.
3. **Higher steady-state quality.** Better-tuned LR → typically 0.3-0.7%
   better final perplexity → 1-2 MMLU points.
4. **Defensible disclosure.** "Hyperparameters tuned via muTransfer wind
   tunnel" is a standard line in modern model cards. Phi-4, MiniCPM,
   Qwen 2.5 all cite this.

## What muP DOES NOT do

- Doesn't change the architecture (same Llama-style decoder).
- Doesn't change training data or token budget.
- Doesn't replace distillation, WSM, doc-masking, QK-norm — these are
  orthogonal and all stack additively.
- Doesn't help if our pilot signal is bad — we still need the pilot to
  validate the full stack end-to-end.
