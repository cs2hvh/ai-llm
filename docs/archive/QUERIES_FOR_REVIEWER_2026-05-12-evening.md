# Queries for Reviewer — 2026-05-12 (evening)

**Subject:** Following your morning review, I ran the 1B-shape benchmarks you asked for and immediately ran into the memory wall you predicted. Below: measured numbers, what an external agent fleet (researching independently) said, and four specific decisions where I need your call before committing to a 1T plan.

**TL;DR:** At 1B/seq=8192 with grad-ckpt + chunked-CE on 5×H200, I'm seeing ~8.7K tok/sec/device and 3.3% MFU. Industry comparables hit 38-56% MFU. The gap is mostly **DP-replicated state + 131k vocab + grad-ckpt forced by OOM**, not a kernel bug. Closing the gap requires FSDP-in-JAX (8-16 hours of work, would save ~$14K on 1T). Question is whether to land that, train at seq=4096 instead, or change the plan.

---

## 1. Updated measured numbers

| Run | Hardware | Config | tok/sec/dev | MFU* | Peak HBM | 1T extrap (5×H200) |
|---|---|---|---|---|---|---|
| pilot baseline (your morning bench) | 1×H200 | 250M @ seq=4096, mb=4 | 47K | 3.4% | 104 GB | ~280 days (wrong; my linear extrap was naive) |
| 1B @ seq=4096, no grad-ckpt | 1×H200 | mb=1 | **15.1K** | 5.7% | 91 GB | ~150 days |
| 1B @ seq=8192, grad-ckpt | 1×H200 | mb=1 | **8.7K** | 3.3% | 61 GB | ~267 days |
| 1B @ seq=8192, grad-ckpt, mb=2 | 1×H200 | doubled batch | 8.3K (basically flat) | 3.1% | 95 GB | ~278 days |
| 1B @ seq=8192, mb=5 across 5 GPUs | 5×H200 | mb=1/device | **OOM** at 140 GB/dev | — | — | — |

*MFU computed against H200 sparse peak (1979 TFLOPS). Dense peak is 989 TFLOPS — so dense MFU is ~2× these numbers (still 6-12%, still well below industry).

**Conclusions from these:**
1. The 250M baseline is the same on 5×H200 (47K) as on 1×H200 (matches my prior bench).
2. Going 250M → 1B drops tok/sec/dev by ~3.1× at constant seq (linear extrapolation was wrong — 5× FLOPs/token).
3. seq=4096 → seq=8192 drops another ~1.8× (attention dominates even with flash; FFN intermediates also grow).
4. Doubling batch on a single GPU **does not improve MFU** at this scale — we're not tensor-core-limited; we're memory-bandwidth-limited.
5. Multi-GPU DP-replicated hits a cliff at 140 GB/device on 5×H200 — NCCL + XLA scratch + preallocator pushed us just over the 141 GB ceiling.

---

## 2. What an external research agent fleet found

I spawned three independent research agents (general-purpose LLM agents with web access) to investigate (a) what published 1B trainers actually achieve, (b) what FSDP/ZeRO would do for our specific config, (c) is our 60 GB / 3% MFU pathological. Their (cross-referenced) findings:

### Industry reference table (verified primary sources)

| Model | Vocab | Seq | tok/sec/dev | MFU | Parallelism |
|---|---|---|---|---|---|
| TinyLlama 1.1B | 32k | 2048 | **24K (A100)** | **56%** | FSDP, no grad-ckpt |
| OLMo 2 32B | ~100k | 4096 | ~1.8K (H100) | **~38%** | FSDP / ZeRO-3 |
| Phi-3-mini 3.8B | 32k | 4096 | ~10.4K (H100, derived) | ~25-30% | not disclosed |
| hackbot.dad 1B Llama-3.2 (home pretrain) | not stated | 4096 | implied ~57K/H100 | **40% multi-GPU** | DDP + flash-attn-2 |
| Llama 3.2 1B (Meta) | 128k | **8192** | NOT DISCLOSED | NOT DISCLOSED | 4D parallelism (DP+TP+PP+SP) |

**Nobody publicly trains 1B at vocab=131k AND seq=8192 simultaneously.** The closest in spirit is Llama 3.2 1B but Meta doesn't disclose throughput. Most 1B trainers go vocab=32-50k + seq=2048-4096.

### Memory math (verified against `gist.github.com/Quentin-Anthony` + Korthikanti et al. + DeepSpeed ZeRO paper)

For our 1.24B-param config:
- bf16 weights: 2.48 GB
- fp32 master: 4.96 GB
- fp32 AdamW (m + v): 9.92 GB
- bf16 grads: 2.48 GB
- **Model+opt subtotal: 19.84 GB/device** (DP-replicated)
- + Activations (grad-ckpt boundary): ~0.6 GB
- + Per-block recompute peak: ~10-15 GB
- + LM-head logits (full vocab): 4.3 GB
- + XLA scratch / NCCL buffers: 5-10 GB
- **Realistic per-device peak: 40-50 GB**

My measured 61 GB is **on the high side but explainable**. Going multi-GPU adds NCCL collective buffers (5-10 GB) + JAX preallocator artifact (the 75% default mem fraction reports 100+ GB usage even when live tensors are smaller).

### FSDP/ZeRO would unlock the math

Same agent computed: ZeRO-3 across 5 GPUs sharding everything (weights, grads, opt state) would cut model+opt from 19.84 → **~4 GB/device**. Per-device peak drops to ~16-25 GB at mb=1, leaving ~115 GB headroom. **mb=4-6 per device at seq=8192** becomes achievable. 3-4× aggregate throughput, ~$14K saving on 1T run.

Implementation in JAX: ~100-150 LOC change to `mesh.py` + `state` init + `train_step`. References: MaxText, Levanter, Keras 3 `LayoutMap` + `NamedSharding`. **8-16 focused hours**.

---

## 3. One silent-bug candidate the agent flagged

`optax.adamw(...)` in [src/myllm/training/optimizer.py:124,151](../../src/myllm/training/optimizer.py#L124-L151) does **not** pass `mu_dtype=jnp.float32` or `nu_dtype=jnp.float32`. Optax default is "same dtype as parameter". With keras mixed_precision="bfloat16", parameters might be stored fp32 (master) but the optimizer state might still be inferred as bf16. **Need to verify**: is our Adam second-moment fp32 or bf16?

If it's bf16, that's:
- A **memory-math error**: my 19.84 GB figure overestimates by ~10 GB
- A **training-quality concern**: bf16 second moment can underflow to zero on small gradients; production trainers typically force fp32

Easy to check: `print(jax.tree.map(lambda x: x.dtype, state["opt_state"]))` after init. Will do this as part of pre-1T checklist regardless of your verdict.

---

## 4. Decisions where I need your judgment

**Q1. Do I implement FSDP-in-JAX before the 1T run?**
- Effort: 8-16 hours (per agent estimate)
- Reward: ~$14K saving on 1T compute (assuming the rest of the plan is fixed) + permanent infrastructure benefit for v2/v3
- Risk: introduces a new parallelism path that needs its own L2/L3 validation; could push 1T start by 3-4 days for testing
- **My leaning:** Yes — the 2-3× throughput win is bigger than any other lever we have, and we'll need it for v2 (3T or 9T tokens) anyway

**Q2. If we don't do FSDP, do we drop to seq=4096 for v1 base, or keep seq=8192 with grad-ckpt + accept ~270 days?**
- seq=4096 at 15K tok/sec/dev: 1T in ~150 days, $65K
- seq=8192 at 8.7K tok/sec/dev + grad-ckpt: 1T in ~267 days, $117K (and we still can't run multi-GPU until we fix the 140 GB ceiling)
- **My leaning:** seq=4096 if we don't do FSDP. Extend to 8192 via YaRN at end-of-pretrain.

**Q3. Do we change the model size?**
- Some agents suggested 500M would fit at mb=2-4/device without grad-ckpt → likely better MFU
- Tradeoff: distillation gradient from Olmo-3-32B teacher is still strong at 500M; final eval scores would be lower but ~Chinchilla-correct
- **My leaning:** Stay at 1B; the project commitment is to 1B per model card

**Q4. How long should the base run actually be?**
- Current plan: 1T tokens (Llama 3.2 style; ~30× Chinchilla-optimal)
- Your morning verdict said "1-5B 1B-shape rehearsal first, then 30-50B 250M pilot, then optionally 100B if curves still reveal uncertainty"
- That maps to a much shorter target than 1T; my 1T plan was over-ambitious for this hardware
- **My leaning:** Adopt your staged plan. 30-50B pilot (250M) → confirm + scale → 100-300B base (1B). Defer 1T to a v2 if results justify it.

---

## 5. What I'm doing in the meantime

Tonight, in parallel with waiting for your reply:
1. Try `XLA_PYTHON_CLIENT_MEM_FRACTION=0.92` on the 5-GPU 1B/seq=8192 bench — zero-effort, may directly fix the OOM
2. Verify Optax mu/nu dtype (the silent-bug candidate)
3. Clean up stale docs/* and consolidate to the latest plan
4. Sketch (not implement) the FSDP-in-JAX migration so I can start fast when you give the go-ahead

Will not start FSDP implementation, will not commit to a 1T plan, will not start a base run, until you respond.

— harshit

---

## Sources cited by agent fleet (so you can verify any claim)

- [Llama 3.2 1B HF card](https://huggingface.co/meta-llama/Llama-3.2-1B)
- [SmolLM2 1.7B](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B), [paper](https://arxiv.org/abs/2502.02737)
- [TinyLlama 1.1B GitHub](https://github.com/jzhang38/TinyLlama)
- [OLMo 2 32B blog (38% MFU)](https://allenai.org/blog/olmo2-32b)
- [Phi-3 paper](https://arxiv.org/html/2404.14219v1)
- [hackbot.dad 1B at home (40% MFU on 8×H100)](https://hackbot.dad/writing/pretraining-at-home/)
- [DeepSpeed ZeRO paper](https://arxiv.org/abs/1910.02054)
- [Korthikanti et al. — Reducing Activation Recomputation](https://arxiv.org/pdf/2205.05198)
- [JAX scaling book](https://jax-ml.github.io/scaling-book/transformers/)
- [Apple Cut Cross-Entropy](https://github.com/apple/ml-cross-entropy) — fused linear-CE kernel
- [Quentin-Anthony memory gist](https://gist.github.com/Quentin-Anthony/f43939791a7ceb0b01a4937308317be5)
