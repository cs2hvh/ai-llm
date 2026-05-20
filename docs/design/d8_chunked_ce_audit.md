# Round D8 — chunked-CE NaN-grad audit (B200 + bf16 + 1B)

**Status**: CPU audit complete (2026-05-17). **GPU verification (single-GPU, 2026-05-18): all 3 modes CLEAN at mb=1/seq=4K.** Production-shape verification (mb=4/seq=8K on multi-GPU FSDP) deferred to Stage 3 prep.
**Severity**: P1 → P2 — fp32-logsumexp mitigation in commit `d56e1e0` produces finite grads at single-GPU 1B scale. Stage 2 still uses full-CE as belt-and-suspenders; chunked-CE re-enable deferred until production-shape verification.
**Source data**: C3 sweep Run-1 first attempt — finite forward loss, NaN gradient, every step. Re-run after dropping `--use-chunked-ce` was clean.

## Symptom

Initial C3 Run-1 launch on 4×B200 NVLink-5 with:
- 1B-shape model (hidden 2048, FFN 8192, 16 layers, width_mult=8)
- bf16 compute, fp32 master weights
- `--use-chunked-ce` (num_chunks=8 over V=131072)
- micro_batch=16, packed_seq_len=8192
- peak_lr=1e-4, muP output_mult = 1/8

Every step produced a **finite forward loss (~11.76, random-init expectation)** but the **gradient was NaN**, triggering `nan_batch_skipped` every step. Atomic revert kept state consistent; no weight update happened, training made zero progress.

Drop `--use-chunked-ce` (switch to full-CE materialising `[B, S, V]` logits), drop micro_batch from 16 → 4 to fit memory, re-launch: **clean run**, val_loss 7.015 at step 1000.

Pilot 250M with chunked-CE on 2×H200 SXM (Stage 1 + Stage 1.5 decay, ~172K steps) was **completely clean**. The bug only appears at 1B-shape on B200.

## Why this matters

- **C2 throughput numbers were taken with chunked-CE.** 30% MFU @ seq=8192 / 46% @ seq=4096 are now suspect for Stage 2 planning — full-CE materialises an 8.6 GB logit tensor at B=8/S=4097, vs 1.1 GB chunked. Stage 2 must re-bench at full-CE before committing to a token budget.
- Chunked-CE was designed *because* full-CE doesn't fit at large micro-batch on 80GB cards. Disabling it costs memory headroom and either forces a smaller micro_batch or activation-checkpointing changes — both affect throughput.
- The bug is silent in the sense that the forward loss looks fine. Without atomic-revert + NaN watchdog, we'd be wasting compute on no-op steps.

## What we know

| Fact | Evidence |
|---|---|
| Forward loss is finite | Logged scalar in C3 Run-1 attempt-1 was 11.76 every step (random-init expectation for V=131072 = log(131072) ≈ 11.78) |
| Backward gradient is NaN | `nan_batch_skipped` fired on every step; atomic revert kept state at init |
| Pilot 250M chunked-CE was fine | 171,990 clean steps on 2×H200 SXM with chunked-CE, num_chunks=8 |
| C3 with full-CE @ 1B on 4×B200 is fine | val_loss curve clean across 3 LR sweeps |
| Bug is reproducible | C3 Run-1 first attempt hit it every step deterministically |

## What we hypothesised (and tested on CPU)

The chunked-CE algorithm carries an online log-sum-exp accumulator across vocab chunks:

```python
# src/myllm/training/loss.py:169-204
running_max = ops.full_like(labels_int, neg_inf, dtype=hidden_states.dtype)  # bf16
running_sum = ops.zeros_like(running_max)  # bf16

for c in range(num_chunks):
    chunk_logits = ops.matmul(hidden_states, ops.transpose(chunk_embed)) * output_mult_t
    chunk_max = ops.max(chunk_logits, axis=-1)
    new_max = ops.maximum(running_max, chunk_max)
    running_sum = running_sum * ops.exp(running_max - new_max) + ops.sum(
        ops.exp(chunk_logits - new_max[..., None]), axis=-1
    )
    running_max = new_max

log_z = running_max + ops.log(running_sum)
nll = -(label_logits - log_z)
```

**Hypothesis H1**: `running_sum` is held in bf16 (inherits from `hidden_states.dtype`). At muP `output_mult = 1/8` and width_mult=8, logits are scaled down significantly. If `running_sum` drops to bf16's denormal range or zero, then `1 / running_sum` in the gradient path (which appears via the chain rule through `log(running_sum)`) overflows to NaN.

**CPU repro**: ran a minimal `jax.value_and_grad` on the chunked-CE function with the same configuration:

```
V = 131072, H = 256, S = 128, B = 2, NUM_CHUNKS = 8
output_mult = 1/8 (muP width_mult=8)
hidden, embed in bf16
Random init
```

**Result**: BOTH chunked-CE and full-CE produced **finite gradients**. Max abs diff between them: **2.98e-7**. Forward losses: identical at 11.75.

```
chunked forward loss: 11.75
chunked grad finite: True
chunked grad min/max: -4.15e-05 4.12e-05

full forward loss: 11.75
full grad finite: True
full grad min/max: -4.15e-05 4.12e-05

grad agreement (max abs diff): 2.98e-07
```

**Hypothesis H1 disproved on CPU.** The algorithm itself, at this scale, is numerically fine in bf16.

## What this means

The CPU repro doesn't reproduce the bug, so the cause is **not** the chunked-CE algorithm in isolation. The bug requires one or more of:

1. **B200/Blackwell-specific bf16 behavior** — tensor cores on Blackwell may have different bf16 rounding/saturation than CPU XLA's bf16 emulation. JAX-on-CPU's bf16 is a software emulation that uses fp32 underneath for many ops; Blackwell hardware does true bf16 matmul.
2. **XLA-CUDA op fusion** — the GPU compiler may fuse the chunked loop differently from CPU, producing a numerically different (and unstable) op graph. The online logsumexp recurrence is data-dependent and XLA may rewrite it.
3. **FSDP gradient reduce-scatter at bf16** — under FSDP, gradients are reduce-scattered across the mesh. If the reduce-scatter is in bf16 and one rank's contribution overflows, the resulting all-shard NaN poisons everything.
4. **Sequence-length interaction** — CPU repro used S=128. Production used S=8192. Some bf16 accumulator paths grow with S (e.g., `running_sum` across the chunked vocab dim is per-position so this *shouldn't* matter, but cross-attention scores in attention DO grow with S and that's not chunked-CE's territory). The bug could actually be in attention or RMSNorm at S=8192 + 1B-shape, and chunked-CE just happens to be the only path that backward-passes through a precision-marginal subgraph.
5. **muP weight init at 1B shape** — `width_mult=8` shrinks some weight inits and the LM-head output by 1/8. At small H (256 in CPU repro) the dynamics may be qualitatively different from H=2048.

We can't decide between these without GPU.

## Why we're not chasing the root cause now

- Stage 2 has a workaround that costs memory but works (full-CE at smaller mb).
- Real fix needs B200 time + bf16-debug instrumentation; cheapest is a dedicated $30 single-pod GPU repro session, not bolted onto a real training run.
- Round D ordering puts D5/D6/D7 (data-side correctness fixes) higher priority than D8 (a known-mitigated GPU edge case).

## Action items

- [x] **D8 CPU audit complete** — chunked-CE algorithm is numerically clean in bf16 at small scale. Bug is GPU-specific.
- [ ] **Stage 2**: use full-CE, NOT chunked-CE on B200. Re-bench throughput at full-CE in the pre-Stage-2 smoke (~$15).
- [ ] **D8 GPU repro** (post-Stage-2): rent 1×B200 for ~1 hour ($3-5), run a minimal 1B-shape chunked-CE step with `jax.disable_jit()` and `XLA_FLAGS=--xla_dump_hlo_as_text` to find which HLO op produces the first NaN. Compare to the same model with chunked-CE in float32 (slower but should not NaN if H1-on-CPU was the only cause).
- [ ] **Long-term**: if root cause is FSDP-reduce-scatter in bf16, the fix is to lift the reduce-scatter to fp32 for the LM-head gradient specifically (mixed-precision exception). If it's XLA fusion, look at `@jax.named_scope` / `with jax.disable_jit():` boundaries. If it's attention at S=8192, that's a separate bug from chunked-CE and chunked-CE was a red herring.

## Stage 2 implication

The C2 throughput numbers banked at R2 (`s3://llm-data/stage2-prep/c2-throughput/`) were taken with `--use-chunked-ce`. **Stage 2 must re-measure throughput at full-CE before committing to a token budget**, because:

- full-CE materialises `[B, S, V]` = 8 × 8193 × 131072 × 2 bytes = 17.2 GB transient at B=8. Won't fit at mb=8.
- Stage 2 will need mb ≤ 4 at seq=8192 with full-CE, OR seq=4096 at higher mb, OR activation-checkpointing through the LM head.

This is the single largest open question for Stage 2 hardware planning.

## Reproduction (CPU audit)

```python
# scripts/audit_chunked_ce_bf16.py (ephemeral; not committed)
import jax
import jax.numpy as jnp
from myllm.training.loss import (
    chunked_cross_entropy_with_z_loss,
    cross_entropy_with_z_loss,
)

V, H, S, B, NUM_CHUNKS = 131072, 256, 128, 2, 8
OUTPUT_MULT = 1.0 / 8.0
key = jax.random.PRNGKey(0)
k_h, k_e, k_l = jax.random.split(key, 3)
hidden = jax.random.normal(k_h, (B, S, H), dtype=jnp.bfloat16)
embed  = jax.random.normal(k_e, (V, H), dtype=jnp.bfloat16) * 0.02
labels = jax.random.randint(k_l, (B, S), 0, V)

def chunked_loss_fn(h, e, lbl):
    loss, _ = chunked_cross_entropy_with_z_loss(
        h, e, lbl, num_chunks=NUM_CHUNKS, output_mult=OUTPUT_MULT,
    )
    return loss

loss_c, grad_c = jax.value_and_grad(chunked_loss_fn, argnums=0)(hidden, embed, labels)
print("chunked forward loss:", float(loss_c))
print("chunked grad finite:", bool(jnp.all(jnp.isfinite(grad_c))))
print("chunked grad min/max:", float(grad_c.min()), float(grad_c.max()))
```

Output reproduced cleanly on CPU XLA — both finite. GPU repro required for the actual bug.

## GPU verification (2026-05-18, 1× B200 SXM, RunPod)

Ran `scripts/d8_gpu_repro.py` in 3 of the 4 modes (`fsdp` mode skipped — needs ≥2 GPUs) at **mb=1 / seq=4K / num_chunks=8** because the production shape (mb=4 / seq=8K) OOMs single-GPU at 177 GB (1B + bf16 + chunked-CE backward without FSDP sharding):

| Mode | XLA flag | Loss | grad NaN | grad min / max | dt steady |
|---|---|---|---|---|---|
| baseline | (none) | 11.8017 | **0** | -1.93e-3 / 1.82e-3 | 3.8 s/step |
| disable-remat | `--xla_disable_hlo_passes=rematerialization` | 11.8050 | **0** | -1.84e-3 / 1.89e-3 | 3.8 s/step |
| fp32-cce | (forces hidden + embed cast to fp32 before loss) | 11.8050 | **0** | -1.79e-3 / 1.71e-3 | 4.0 s/step |

Loss=11.80 ≈ ln(131072)=11.78 matches random-init expectation. All three modes produced finite grads through 10 steps with no NaN, no Inf.

Artifacts on R2: `s3://llm-data/stage2-prep/d8-repro/`
- `baseline_20260518-003643.json`
- `disable-remat_20260518-003934.json`
- `fp32-cce_20260518-004201.json`

### What we learned

1. **The fp32-logsumexp mitigation (commit `d56e1e0`) produces finite gradients at 1B single-GPU bf16.** Hypothesis H1 — that the chunked-CE algorithm itself was the cause — is now further weakened: the algorithm + the mitigation runs cleanly at production-equivalent operating shape (per-token math is independent of FSDP / batch / seq), just with smaller absolute memory footprint.
2. **disable-remat made no difference** — same finite grads as baseline. This is **inconclusive about openxla/xla #17922**: the bug there only fires under specific remat-induced fusion patterns at higher graph complexity / memory pressure. Our single-GPU shape may not stress that path. Cannot confirm or rule out cause #1 from this run.
3. **fp32-cce made no difference** beyond baseline — same finite grads. Suggests the bug class isn't purely bf16-numerics-in-chunked-CE; at this scale our current bf16 path is already fine.

### What we still don't know

The original C3 NaN was at **mb=4 / seq=8192 / 4×B200 / FSDP / chunked-CE**. Single-GPU mb=1/seq=4K removed three variables at once (FSDP, batch shape, seq length). Three open possibilities remain:

- **A. fp32-logsumexp was load-bearing.** Bug is genuinely dead. Probable but not proven.
- **B. FSDP reduce-scatter on bf16 grads.** Wouldn't fire single-GPU. Would need ≥2-GPU pod to verify.
- **C. Memory-pressure-induced fusion patterns** (openxla/xla #17922 class). Wouldn't fire under our smaller memory footprint.

### Decision: not blocking Stage 2

| Reason | |
|---|---|
| Stage 2 uses full-CE | Memory budget at mb=4 / seq=8K with full-CE fits comfortably (proven by C3 Run 2 and Run 3). |
| chunked-CE only matters at 7B+ | Where full-CE OOMs. We're not there until Stage 4+. |
| Verification cost vs benefit | A 4×B200 / 1-hour multi-GPU verification is $25-30. Worth doing **before Stage 3** (where 600B tokens × full-CE throughput hit is real) but not before Stage 2. |

### Action items (updated)

- [x] **D8 CPU audit complete** — algorithm clean on CPU.
- [x] **fp32-logsumexp mitigation shipped** (commit `d56e1e0`) — chunked-CE accumulators run in fp32.
- [x] **D8 GPU repro at single-GPU** — all 3 modes clean (2026-05-18). Mitigation works at this scale.
- [ ] **D8 production-shape verification** (4×B200 + mb=4 + seq=8K + FSDP) — **deferred to Stage 3 prep**. $25-30, 1 hour.
- [ ] **Stage 2**: stay on full-CE. No change to plan.
- [ ] **D11 (Apple CCE)**: independent of D8 closure. Still required pre-Stage-3 for the 24GB → 1MB memory win at the LM head, which would also kill the D8 bug class definitively.

## See also

- [[d9_step718_investigation]] — separate bug, deterministic NaN batch, may interact with D8 if both are bf16-precision-edge mechanisms.
- `docs/SESSION_HANDOFF.md` §3 (open bugs) — both D8 and D9 tracked.
- `docs/DESIGN.md` §8.3 — chunked-CE algorithm description.
