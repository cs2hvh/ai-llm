# Architecture Review — v1 (2026-05-10)

**Reviewer:** acting AI-researcher hat. **Subject:** the architecture, hyperparameters, and code I (Claude) shipped in PLAN.md and the `myllm` package. **Method:** compare against authoritative configs of recently-trained small LLMs, plus a code-level self-audit.

This is a "I-found-issues-with-my-own-work" document. The point is to surface them before we spend money.

---

## 1. Comparison against published configs

Pulled from `config.json` files on Hugging Face for ground truth.

| Param | **My pilot 250M (v0)** | **My base 1B (v0)** | Llama 3.2 1B | SmolLM2 1.7B | Qwen 2.5 0.5B | Qwen 2.5 1.5B |
|---|---|---|---|---|---|---|
| layers | 16 | 24 | **16** | 24 | 24 | 28 |
| hidden | 1024 | 2048 | **2048** | 2048 | 896 | 1536 |
| FFN | 2752 | 5504 | **8192** | 8192 | 4864 | 8960 |
| **FFN ratio** | **2.69×** | **2.69×** | **4.0×** | **4.0×** | **5.43×** | **5.83×** |
| heads | 16 | 16 | **32** | 32 | 14 | 12 |
| KV heads | 4 | 8 | **8** | 32 (no GQA) | 2 | 2 |
| GQA ratio | 4:1 | 2:1 | **4:1** | 1:1 | 7:1 | 6:1 |
| head_dim | 64 | 128 | **64** | 64 | 64 | 128 |
| vocab | 128k | 128k | **128k** | 49k | 152k | 152k |
| tie embeds | yes | yes | **yes** | yes | yes | yes |
| **RoPE base** | **10k** | **10k** | **500k** | 130k | 1M | 1M |
| RMS eps | 1e-5 | 1e-5 | 1e-5 | 1e-5 | 1e-6 | 1e-6 |
| ctx (train) | 4k | 4k | 8k native, 128k extended | 2k → 8k | 32k | 32k → 128k |
| **train tokens** | 30B | 300B | **9T** | 11T | ~12T | ~18T |
| **tok/param** | 120 | 230 | **7250** | 6500 | ~24,000 | ~12,000 |

(Bold = where my v0 differs materially from current SOTA practice.)

## 2. Findings

### 🔴 Critical (will hurt model quality)

**F1. FFN ratio of 8/3 (2.69×) is outdated.**
- Origin: Llama 1 (2023) used 8/3 because of the SwiGLU param-count adjustment.
- Modern recipe: **4× for Llama 3.x and SmolLM2; 5–6× for Qwen 2.5**.
- 4× gives more capacity per layer at the cost of more FLOPs/token; the bias-variance tradeoff has shifted toward 4× since 2024.
- **Fix:** bump pilot FFN to 4× (1024 → 4096 if hidden stays, or restructure), bump base FFN to 4× (2048 → 8192).

**F2. Token budget is severely under-trained vs modern practice.**
- 300B tokens for 1B = 230 tok/param, near Chinchilla compute-optimal (~20:1 for 70B but lower for inference-deployable smalls).
- Modern small-LLM practice: **6,500–24,000 tok/param.** SmolLM2 1.7B at 6500:1, Qwen 3 0.6B at 60,000:1.
- At our $15M budget, training 1B for 1–3T tokens is well within reach (~$200k–$600k of 1B's training cost).
- **Fix:** revise PLAN.md token targets. Bare-min 500B → strong 1T → ambitious 3T.

**F3. RoPE base = 10000 is outdated.**
- 10k was correct for 2K-context Llama 1.
- Modern 8K-targeting models use **130k+ (SmolLM2)**, **500k (Llama 3.2)**, or **1M (Qwen)**.
- Higher base gives the model more low-frequency components, helps long-context generalization.
- **Fix:** pilot RoPE base 130k, base RoPE base 500k.

**F4. `train_step.py` calls `model.stateless_call` with the wrong signature.**
- I wrote `model.stateless_call(params, batch["input_ids"])`.
- Correct Keras 3 signature: `model.stateless_call(trainable_variables, non_trainable_variables, inputs)` returning `(outputs, updated_non_trainable_variables)`.
- Our model has non-trainable variables (RoPE cos/sin tables) — the current code ignores them.
- **This is a runtime bug; would fail on first JIT trace.**
- **Fix:** rewrite train_step to thread (trainable, non_trainable) properly.

### 🟡 Important (shipping this is OK but not enterprise-grade)

**F5. Attention not using FlashAttention / Pallas SDPA.**
- My attention computes `softmax(QK^T/√d) @ V` explicitly. Materializes the [seq, seq] attention matrix.
- For 4K context, the matrix is 64MB/head/batch in fp32 — borderline; for 8K it's 256MB/head/batch — bad.
- Production move: route through `jax.numpy.dot_product_attention` (calls Pallas SDPA when on GPU/TPU).
- **Fix later:** wrap attention in a backend-aware fast path; defer until we hit perf wall in pilot.

**F6. No sharding annotations on weights.**
- For multi-GPU/multi-node FSDP, parameters need `jax.sharding.NamedSharding(mesh, P("model", None))` annotations.
- Current code is correct for **data-parallel only**. 1B on 32 GPUs requires FSDP.
- **Fix later (pre-Phase-4):** add sharding-spec module that walks the param PyTree and annotates.

**F7. Init is uniform N(0, 0.02). Llama uses scaled init for residual projections.**
- Output projections (`wo`, `w_down`) get N(0, 0.02 / √(2L)) in Llama-style.
- Mitigates late-training divergence by keeping residual stream bounded.
- **Fix:** add `scaled_init` flag to layers.py, default off for pilot, enable for 1B base.

**F8. No QK normalization.**
- Llama 3 and Gemma 2 use QK-norm (RMSNorm on Q and K before attention) for training stability at scale.
- Documented as deferred in plan.
- **Fix later:** add behind a flag, decide based on pilot loss curve.

**F9. Stack choice (Keras 3 + JAX) is uncommon for from-scratch pretraining.**
- Production examples in 2025–26: TorchTitan/PyTorch FSDP, MaxText/Picodo/JAX-AI-Stack (pure JAX/Flax), Megatron-LM. **None I found use Keras 3 + JAX as the from-scratch pretraining loop.**
- The model code is fine (`keras.ops` is portable), but the training-loop ergonomics for SPMD via stateless_call are clunkier than pure Flax.
- **Decision:** keep Keras 3 + JAX for now. Document the alternative (pure JAX/Flax via Picodo or MaxText style). Reconsider if pilot reveals friction.

### 🟢 Confirmed correct

- Decoder-only Llama-style ✅
- RoPE (interleaved-pair form is mathematically equivalent to Llama's split-half for from-scratch training) ✅
- RMSNorm with pre-norm placement ✅
- SwiGLU FFN with three projections ✅
- Tied embeddings for small-vocab/large-embed model ✅
- AdamW β=0.9/0.95, WD 0.1, grad-clip 1.0 ✅
- z-loss 1e-4 (PaLM/Llama 2 standard) ✅
- LR peaks 3e-4 (pilot), 2e-4 (base); cosine with 2k warmup ✅
- Vocab 128k matches Llama 3.x ✅
- bfloat16 weights/activations + fp32 optimizer state ✅
- MinHash+LSH params (128 perm, 32 bands × 4 rows, threshold 0.85) — within standard range ✅

## 3. Recommended revisions

### Pilot 250M → "MyLLM-Pilot-250M-v1"

| Param | v0 | **v1 (new)** | Source of v1 choice |
|---|---|---|---|
| layers | 16 | **16** | unchanged |
| hidden | 1024 | **768** | sized for 250M with 4× FFN + 128k tied vocab |
| FFN | 2752 | **3072** | 4× ratio |
| heads | 16 | **12** | head_dim 64; same dim as hidden / head_dim |
| KV heads | 4 | **4** | 3:1 GQA |
| head_dim | 64 | **64** | unchanged |
| RoPE base | 10000 | **130000** | SmolLM2-style for 8k context |
| **est. params** | ~250M | **~249M** | recomputed |

### Base 1B → "MyLLM-Base-1B-v1" (matches Llama 3.2 1B exactly)

| Param | v0 | **v1 (new)** | Source of v1 choice |
|---|---|---|---|
| layers | 24 | **16** | matches Llama 3.2 1B (wider, fewer layers) |
| hidden | 2048 | **2048** | unchanged |
| FFN | 5504 | **8192** | 4× ratio (Llama 3.2 1B) |
| heads | 16 | **32** | matches Llama 3.2 1B |
| KV heads | 8 | **8** | unchanged (4:1 GQA) |
| head_dim | 128 | **64** | matches Llama 3.2 1B |
| RoPE base | 10000 | **500000** | matches Llama 3.2 1B |
| **est. params** | ~1.37B | **~1.24B** | recomputed |

**Why match Llama 3.2 1B exactly:** the architecture is proven; "sovereign" is about owning the tokenizer + weights + training data, not about a bespoke architecture. We can iterate on architecture in v2.

### Token budget bump in PLAN.md

| Plan section | v0 | **v1** |
|---|---|---|
| Pilot tokens | 30B (baseline) / 50B (stretch) | **50B (baseline) / 100B (stretch)** |
| Base 1B bare-min | 100B | **500B** (still under-trained but minimum viable) |
| Base 1B strong | 300B | **1T** (matches modern small-LLM practice at 1000:1) |
| Base 1B ambitious | 500B | **3T** (Qwen-2.5-class) |

Cost implications (still well within $15M ceiling):
- Pilot 50B: ~$5k
- Base 1B at 1T: ~$200k–300k
- Base 1B at 3T: ~$600k–900k

## 4. Code-audit findings (to fix in this commit)

| ID | File | Issue | Fix |
|---|---|---|---|
| F4 | `training/train_step.py` | wrong stateless_call signature | rewrite to thread (trainable, non_trainable) |
| F1+F3 | `configs/pilot_250m.yaml`, `configs/base_1b.yaml` | outdated FFN ratio + RoPE base | revise to v1 spec above |
| F7 | `model/layers.py` | init not scaled for residual paths | add `scaled_init_for_residuals` flag |
| F2 | `PLAN.md` §7 | undertrained token target | bump to v1 token budget |

## 5. Items deliberately deferred

| ID | Item | Why defer |
|---|---|---|
| F5 | FlashAttention via Pallas | not needed at pilot scale; profile first |
| F6 | Sharding annotations | needed pre-Phase-4 base run, not Phase-2 pilot |
| F8 | QK norm | decide based on pilot loss-curve health |
| F9 | Stack swap to pure JAX/Flax | would invalidate model code; revisit if pilot is painful |
| — | Decontamination module | Phase-3 work, not Phase-0 |
| — | KenLM quality classifier | Phase-3 work |
| — | Language-ID filter | Phase-3 work |

## 6. What I am *not* changing

- Keras 3 + JAX backend choice — same reasoning as before, not flipping mid-stream.
- 7-language tokenizer mix — design is sound, blockers are corpus-availability not arch.
- Data pipeline filters — heuristics are reasonable defaults, refine on real corpus stats.
- RunPod orchestration — not architectural.

---

*This review supersedes the architecture parts of v0 PLAN.md. Changes applied in the same commit.*
