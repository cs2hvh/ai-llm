# muP / muTransfer Design — v1 (2026-05-11, revised 2026-05-12)

**Decision owner:** harshit.hv@samatva.com
**Status:** ✅ SHIPPED (model multipliers + per-param LR via `optax.multi_transform` + Proxy A sweep wired). Proxy B 300M transfer validation pending. R1 from `docs/ai_research_dossier_2026-05-11.md`.

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

For our current configs (`mup.base_width: 256` throughout):

| Config | hidden_dim | width_mult vs base=256 | Param count |
|---|---|---|---|
| Wind-tunnel **Proxy A** (`configs/wind_tunnel.yaml`) | 384 | 1.5 | ~67M |
| Wind-tunnel **Proxy B** (`configs/wind_tunnel_b.yaml`) | 1024 | 4.0 | ~300M |
| Pilot (`configs/pilot_250m.yaml`) | 1024 | 4.0 | ~250M |
| Base 1B (`configs/base_1b.yaml`) | 2048 | 8.0 | ~1.24B |

Proxy B was added 2026-05-12 per external reviewer's recommendation: validate
that the LR/init optimum found on Proxy A actually transfers as predicted
**before** committing to the 1B base run (which costs $11-25K).

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

Once the three multipliers + per-param-LR are in place (✅ done):

1. **Proxy A** (`configs/wind_tunnel.yaml`): hidden=384, width_mult=1.5, ~67M params, context 2048.
2. Sweep over **5 LR values × 2 init scales = 10 cells** (see `scripts/wind_tunnel_sweep.py` + `tests/test_wind_tunnel.py`).
3. Each cell trains for **200M tokens** (dropped from initial 1B plan per μP literature — 200M is enough to see a clean U-curve). ~$3-5/cell, ~$30-50 total sweep.
4. Pick the LR / init that gives the lowest end-loss.
5. **Proxy B** (`configs/wind_tunnel_b.yaml`): width_mult=4.0, ~300M params, one cell at the chosen (LR*, init*) for 500M tokens. Verifies the transfer law before pilot/base. ~$11-20.
6. **Apply to pilot 250M (m=4) and base 1B (m=8)** — no further tuning.

The "zero-shot transfer" claim is that the LR which is optimal at 67M is
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

## File touch list (status as of 2026-05-12)

All shipped:
- [x] `docs/mup_design.md` — this file
- [x] `src/myllm/model/config.py` — `MupConfig` with `apply_*_output_mult` ablation knobs
- [x] `src/myllm/model/layers.py` — output multipliers in `GroupedQueryAttention` + `SwiGLUFFN`
- [x] `src/myllm/model/transformer.py` — LM-head output multiplier
- [x] `src/myllm/training/optim.py` — per-parameter LR scaling via `optax.multi_transform` (B1 fix: state restored as `MultiTransformState` namedtuple)
- [x] `configs/wind_tunnel.yaml` — Proxy A (67M, width_mult=1.5)
- [x] `configs/wind_tunnel_b.yaml` — Proxy B (300M, width_mult=4.0) — added 2026-05-12 per reviewer
- [x] `configs/pilot_250m.yaml` + `configs/base_1b.yaml` — `mup.base_width: 256` set
- [x] `scripts/wind_tunnel_sweep.py` — 10-cell sweep launcher (with `--peak-lr-override`, `--init-std-override`, `--micro-batch-override`)
- [x] `tests/test_mup.py`, `tests/test_wind_tunnel.py` — width-invariance + sweep regression tests

Pending:
- [ ] Proxy A sweep execution (sweep terminated 2026-05-12 per user direction; pending re-launch)
- [ ] Proxy B single-cell transfer validation
- [ ] Pilot 250M launch using the validated (LR*, init*)

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
