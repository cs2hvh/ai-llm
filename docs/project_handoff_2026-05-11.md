# MyLLM — Project Handoff & Review Brief
*Snapshot date: 2026-05-11 (revised: 2026-05-11 PM)*

---

## TL;DR

We are building **MyLLM**, a from-scratch 1B-parameter decoder-only foundation model, **English-primary with sovereign Indic hedge** (Hindi sub-mix at ~4% pretrain, ~20% tokenizer). Single-person engineering, treated as an enterprise project not a learning exercise. Phase 1 (tokenizer) shipped; we are mid-Phase 2 (proxy HP sweep on a 67M wind-tunnel model under muP), with Phase 3 (base 1B at 1T tokens) gated on the sweep result.

**As of 2026-05-11 PM**: spent ~5 hours debugging a deterministic NaN at training step 4200 in the proxy sweep. Systematically isolated it: tried 4 different LR/init/seed combinations (all NaN'd at step 4200), then ran with `--synthetic-data` which ran cleanly through step 6000. **Conclusion: the NaN is data-driven — a specific batch of FineWeb-Edu docs at ~step 4200 produces a NaN forward pass.** Applied a standard NaN-skip patch in `train_step.py` (zero out gradients on bad-batch steps); sweep is restarting now. Open question for you: whether NaN-skip alone is sufficient, or whether we should also identify and filter the offending docs.

Otherwise the stack is intact: 256 unit tests pass, distillation infra is wired end-to-end (synthetic teachers only so far), R6 decontamination is plumbed, eval harness has MMLU-ProX + Belebele + MILU adapters.

We'd value an outside review of: (a) the architecture-and-training-recipe choices, (b) the data mix and gated-source contingency plan, (c) the NaN-at-step-4200 root cause + whether NaN-skip is the right primary defense, and (d) whether the 1T-token plan for 1B params is the right call vs alternatives. Specific questions at the bottom.

---

## 1. Context & Goals

### Mission
Build the **best possible 1B-class English-primary foundation model** that also reads/writes Hindi competently. "Best possible at 1B" — comparable or better than Llama 3.2 1B, SmolLM2 1.7B, and Qwen 2.5 1.5B on MMLU/Belebele/HumanEval, with the additional Indic capability from Sangraha.

### Constraints
- **Solo lead**: harshit.hv@samatva.com is sole engineer + ML researcher.
- **Compute budget flexibility**: not strictly capped, but cost-aware. The wind-tunnel sweep is ~$20-30; pilot is ~$200-500; base 1B at 1T tokens is ~$11-25K depending on SKU (B200 wins, H200 SXM second).
- **No team**: every decision is one person's call. External review (this doc) is how we validate.
- **Open-license posture**: model weights to be released under Apache-2.0 or MIT.

### Success criteria
Phase 2 (pilot 250M):
- Loss curve is sensible (smooth descent, no NaN)
- Watchdog/checkpoint/resume pipeline survives a synthetic crash
- R2 mirror works for checkpoints
- Tokens/sec is within 20% of B200-on-1B baseline (~150K tok/sec target for the bigger model)

Phase 3 (base 1B):
- MMLU ≥ 42% (Llama 3.2 1B parity)
- Belebele ≥ 50% on English, ≥ 35% on Hindi
- HumanEval+ ≥ 15% (modest for a 1B base)
- Distillation "invisible" — model behavior shouldn't read as a thin DeepSeek/Qwen/Mistral wrapper

---

## 2. Architecture decisions (audit-worthy)

### Model
| Choice | Value | Source / Rationale |
|---|---|---|
| Family | Decoder-only Transformer (Llama-3-style) | Industry default at this scale |
| Layers | 16 (base 1B); 8 (proxy 67M) | Scaling layers ∝ √N |
| Hidden | 2048 (base); 384 (proxy) | width_mult = 8 (base) / 1.5 (proxy) under muP |
| FFN ratio | 4× | Llama default |
| Attention | **GQA 3:1** (6 heads, 2 KV per layer at proxy) | KV-cache savings; Mistral/Llama-3 pattern |
| Head dim | 64 | Standard |
| RoPE base | 130,000 | Llama-3 default; long-context-friendly |
| Norm | RMSNorm, eps=1e-5 | Llama default |
| Activation | SwiGLU | Llama default |
| **QK-norm** | **Yes, post-RoPE** | Llama-3 convention; mP transfer expects this |
| **Tied embeddings** | **Yes** | Memory savings; common at this scale |
| Vocab | 131,072 (2^17, SPM-Unigram + byte-fallback, NFKC, Metaspace) | Phase 1 tokenizer, shipped |
| Z-loss | coef = 1e-4 | Stability anchor |
| Grad clip | global_norm=1.0 | Conservative |

**Open question**: GQA 3:1 might be too aggressive at 1B scale (vs Llama 3.2 1B's 8:2 = 4:1). We may want to revisit at Phase 3.

### muP / muTransfer ([Yang et al., Tensor Programs V](https://arxiv.org/abs/2203.03466))
We use the **EleutherAI "minimal" variant** of muP:
- Per-parameter LR scaling via Optax `multi_transform` (embedding LR unscaled, hidden weights scaled by 1/width_mult)
- Output multipliers on attention output, FFN output, lm_head (configurable)
- Sweep at base_width=256 (proxy hidden=384, width_mult=1.5)
- Transfer to pilot 250M (hidden=1024, width_mult=4) and base 1B (hidden=2048, width_mult=8)

**Bug found just now**: muP's `MultiTransformState` doesn't round-trip through Orbax checkpointing — restored opt_state comes back as a plain dict, breaking `state.inner_states` access. Must fix before Phase 2 pilot (which checkpoints across spot interruptions). On the todo list.

### Schedule: WSD (Warmup-Stable-Decay)
- Warmup: 10% of total steps (linear)
- Stable: 75% (constant peak_lr)
- Decay: 15% (linear to 0.1 × peak_lr)

Rationale: WSD lets you cool-down ANY stable-phase checkpoint to a usable model in 10-15% additional compute. Decouples token budget from final-model decisions.

### WSM (Warmup-Stable-Merge, [arXiv:2507.17634](https://arxiv.org/abs/2507.17634))
Plan to merge the last 3-5 stable-phase checkpoints (uniform weight average) before decay. Reportedly gives a ~1-2% capability boost essentially for free.

### Decay-phase distillation (R0 in our internal dossier)
At step ≥ 0.85 × total_steps, the training loop injects pre-cached top-K logits from three teacher models into each batch. The train_step's mixed loss becomes:

```
loss = α · CE(student, gold) + (1-α) · mean_t KL(softmax(teacher_t_topK) || softmax(student_topK))
```

- α = 0.3 (literature anchor)
- Temperature = 1.0
- top-K = 8
- Three teachers: **DeepSeek-V4-Pro-Base** (MIT), **Qwen 3.6-27B** (Apache-2.0), **Mistral-Medium-3.5-128B** (Apache-2.0)
- All teachers are **base, not chat-tuned**, to preserve "invisibility" (model shouldn't read as any specific teacher)

Top-K logits are cached offline (~$15-25K compute) in Arrow IPC shards, mmap'd at runtime.

### Intra-document attention masking (R2)
We use `segment_ids` threaded through `jax.nn.dot_product_attention` (cuDNN flash-attention backend) so packed sequences don't attend across document boundaries. Standard but not always done at smaller scales.

---

## 3. Data pipeline

### Pretrain mix (target, before gating issues)

| Source | Share | Category | Notes |
|---|---|---|---|
| HuggingFaceFW/fineweb-edu | 31.5% | web | Open, parquet |
| nvidia/Nemotron-CC HQ | 13.5% | web | **GATED**, pending NVIDIA approval |
| bigcode/the-stack-v2 | 18% | code | Gated, T&Cs **accepted** |
| wikimedia/wikipedia | 6% | wiki | Open |
| pg19 | 5% | books | Custom loader, public domain |
| allenai/peS2o | 6% | academic | Custom loader |
| open-web-math/open-web-math | 7% | math | Open, parquet |
| HuggingFaceH4/stack-exchange-preferences | 2% | qa | Custom loader |
| ai4bharat/sangraha (split=hin) | 4% | multilingual (Hindi) | Sovereign hedge — AI4Bharat |
| mc4 (es, zh, ar, fr, de) | 8% | multilingual | Custom loader → redirects to allenai/c4 |

**Total: 100%**. We dropped `EleutherAI/proof-pile-2` (4.2%) after multiple loader-script failures (zstd decompress error mid-stream) — math share absorbed by open-web-math.

### Filters
- Length: 200 chars min, 1M chars max
- Repetition: top-word share ≤ 20%, top-5gram share ≤ 10%
- Symbol ratio: ≤ 30%
- PII: redact email + phone (not IPv4)

### Decontamination (R6)
- 13-gram (Llama-2 / OLMo-2 convention)
- xxhash64
- Indexed against MMLU-ProX + Belebele + MILU prompts at 200 samples/lang each
- OLMo-2-style per-gate CSV report

### Sequence packing
- Pack documents into 2048 (proxy) or 4096 (pilot/base) tokens
- `segment_ids` track document boundaries for intra-doc masking
- No padding wasted; drop incomplete final pack

### Tokenizer (Phase 1, **shipped**)
- SentencePiece Unigram, 131k vocab
- byte_fallback for unseen Unicode
- NFKC normalization, Metaspace pre-tokenizer
- 7 product languages + Indic (Sangraha at 20% during tokenizer training)
- Stored at `s3://llm-data/tokenizer/myllm-spm-unigram-131k-v2.json`

---

## 4. Training methodology summary

```
Phase 1 (DONE):           SP-Unigram tokenizer, 131k vocab, English + 6 secondary + Indic
Phase 2 (IN PROGRESS):    Wind-tunnel HP sweep at 67M proxy
                          → Pilot 250M run (5-10B tokens, $200-500)
                          → Gates Phase 3
Phase 3 (PENDING):        Base 1B, 1T tokens, WSD schedule + WSM merge + decay-phase distillation
                          $11-25K compute (depends on SKU)
                          Eval: MMLU/Belebele/MILU/HumanEval+
Phase 4 (DEFERRED):       SFT + persona-strip + alignment passes
```

We deliberately **do NOT** plan to:
- Train chat-tuned variants in Phase 3 (base only)
- Use RLHF/DPO until SFT is done and judged
- Scale past 1B until 1B is validated

---

## 5. Tech stack

| Layer | Choice |
|---|---|
| Language | Python 3.11 |
| ML framework | **JAX** + **Keras 3** (`KERAS_BACKEND=jax`) |
| Optimizer | Optax AdamW with `multi_transform` for muP per-param LR |
| Checkpointing | Orbax (with R2 mirror via boto3) |
| Distributed | JAX mesh — data-parallel today; FSDP in `myllm.training.mesh` once we go multi-GPU |
| Attention | `jax.nn.dot_product_attention` with cuDNN backend (flash-attention) |
| Tokenization | SentencePiece native binary; `tokenizers` library only for smoke tests |
| Data | HF `datasets` streaming over packing layer (`SequencePacker`) |
| Eval | Custom harness (`myllm.eval.runner`) implementing the `Benchmark` Protocol; lm-eval-harness used for adapter cross-checks only |
| Tracking | Weights & Biases (`wandb` ≥0.18) |
| Compute | RunPod (H100/H200/B200 SKUs); RunPod SDK 1.9 wrapped in our `myllm.runpod_orch` |
| Storage | Cloudflare R2 (S3-compatible) for tokenizer, code drops, checkpoints, teacher caches |
| Testing | pytest, 256 tests passing, ruff for lint |
| CI | Local-only currently; we manually run `pytest` + `ruff` before commits |

---

## 6. Repo layout

```
llm-build/
├── src/myllm/
│   ├── model/            # ModelConfig, layers (GQA, SwiGLU, RMSNorm), TransformerLM
│   ├── training/         # loop.py, train_step.py, optimizer.py (muP), checkpoint.py (WSM),
│   │                     #   watchdog.py, decay_phase.py, schedule.py, mesh.py
│   ├── data/             # types.py, loader.py (HFStreamLoader), filters.py (LengthFilter etc.),
│   │                     #   decontamination.py, mixture.py, pack.py, tokenize.py,
│   │                     #   teacher_cache.py (Arrow IPC reader + MultiTeacherCacheReader)
│   ├── eval/             # types.py (Benchmark Protocol), runner.py, benchmarks/{mmlu_prox,belebele,milu}.py
│   ├── runpod_orch/      # client.py, spec.py (GPUSku enum, hourly $ table), lifecycle.py, cost.py
│   └── utils/            # storage.py (R2 S3 client), exceptions.py, logging
├── scripts/
│   ├── run_pretrain.py             # main training entrypoint
│   ├── wind_tunnel_sweep.py        # 10-cell HP sweep driver
│   ├── cache_teacher_logits.py     # teacher logit producer (synthetic + vLLM-stubbed)
│   ├── build_decontamination_index.py  # offline R6 index builder
│   ├── smoke_test_datasets.py      # preflight: pull 5 rows from every source (NEW)
│   ├── bootstrap_pod.sh            # pip + venv setup on pod
│   └── train_tokenizer_spm.py      # Phase 1 tokenizer trainer
├── configs/
│   ├── base_1b.yaml                # Phase 3 model config
│   ├── pilot_250m.yaml             # Phase 2 model config
│   ├── wind_tunnel.yaml            # 67M proxy config
│   ├── decay_phase_distillation.yaml
│   └── data/pretrain_mix.yaml
├── tests/                          # 26 test files, 256 tests passing
└── docs/
    ├── ai_research_dossier_2026-05-11.md  # internal R0-R8 recommendations
    ├── audit_2026-05-11_session_followup.md
    ├── mup_design.md
    ├── teacher_distillation_strategy.md
    ├── teacher_logit_cache_format.md
    ├── wind_tunnel_launch_checklist.md
    ├── phase2_launch_checklist.md
    ├── playbook_alignment.md
    └── project_handoff_2026-05-11.md    # THIS DOCUMENT
```

---

## 7. Current status (2026-05-11 PM)

### Done
- Phase 1 production tokenizer SHIPPED
- All R0-R8 dossier recommendations implemented:
  - R0: Distillation infra (loss, train_step, cache reader, decay-phase activation)
  - R1: muP scaffolding + per-param LR + wind-tunnel sweep driver
  - R2: Intra-document attention masking
  - R3: QK-norm post-RoPE
  - R4: Nemotron-CC swap (config, pending data access)
  - R5: WSM checkpoint merging
  - R6: N-gram decontamination + pipeline integration + offline index builder
  - R8: Eval harness + MMLU-ProX/Belebele/MILU adapters
- 256 unit tests passing
- Wind-tunnel sweep launched on 1× H200 SXM (RunPod, secure cloud)
- **2026-05-11 PM additions**:
  - Git repository initialized + first commit (`6834eb9`), pending push to GitHub remote
  - `--no-watchdog` CLI flag added to `run_pretrain.py` (sweep cells need to run to completion regardless of spikes; final-loss is the signal we want)
  - `scripts/smoke_test_datasets.py` preflight script added (pulls 5 rows from each source on control plane CPU, surfaces ALL loader bugs before pod launch) — per dossier audit recommendation
  - `--split` threading through `HFStreamLoader` (was hardcoded to "train"; broke ai4bharat/sangraha which uses per-language splits like `split: hin`)
  - `pretrain_mix.yaml` restored after debug session: 13 sources, total=1.0 exactly. Adjustments: dropped Nemotron-CC (gated, pending approval) + FineWeb-Edu absorbs its share at 0.44; dropped proof-pile-2 permanently (4 different failure modes including zstd decompression error); the-stack-v2 restored (T&Cs accepted); sangraha uses `split: hin`
  - **`zstandard` dep added** to `pyproject.toml` + `requirements.txt` (HF proof-pile-2 loader script imports it directly; bootstrap was missing it)
  - **NaN-skip patch added to `train_step.py`** (see "Debug investigation" section below) — zeroes out gradients when loss or any grad is non-finite; bad batches silently skipped, training continues

### In progress
- Wind-tunnel HP sweep relaunching with NaN-skip patch + `--no-watchdog`. Stage 1 (single diagnostic cell, LR=1e-3 init=0.02) already showed clean step 100→3700 loss trajectory before the historical NaN at 4200. The NaN-skip patch should let it survive that spike and complete.

### Open / blocked
- **NaN root cause unidentified** — we know it's a specific FineWeb-Edu batch at step ~4200 (deterministic by streaming order), but we haven't instrumented the loader to dump the actual doc contents. The NaN-skip patch unblocks training but doesn't tell us what's in the bad doc.
- `nvidia/Nemotron-CC` data access pending NVIDIA manual approval (the-stack-v2 already approved)
- Real-teacher integration (vLLM path stubbed; synthetic-teacher path tested)
- R2 lazy-fetch for shards (reader requires pre-downloaded local files)
- **muP `multi_transform` opt_state doesn't round-trip through Orbax** — `MultiTransformState` namedtuple gets restored as a plain dict, breaking `state.inner_states` access. MUST fix before Phase 2 pilot (which depends on checkpoint resumption across spot interruptions). Workaround for the sweep: never resume (checkpoint cadence is sparser than cell length).
- Phase 2 pilot 250M not yet booked
- Phase 3 base 1B not yet booked

### Known issues we already plan to fix
- mc4 is deprecated — should swap to `allenai/c4` parquet-native
- Keras 3 `supports_masking` warning is noisy (cosmetic; the GQA layer doesn't declare `supports_masking=True` but we pass our own mask explicitly so nothing is lost — just spammy log output)
- Data pipeline doesn't pin `revision=` per HF source — non-deterministic across HF history rewrites
- The NaN-at-step-4200 doc still needs identification (instrumented loader → dump batch contents → inspect)

---

## 8. Cost & timeline

| Phase | What | Cost (estimated) | Wall time |
|---|---|---|---|
| Phase 1 | Tokenizer | ~$50 (done) | done |
| Phase 2a | Wind-tunnel sweep (10 cells × 200M tokens, 67M proxy, 1× H200 SXM) | $20-30 | ~5-10 hr |
| Phase 2b | Pilot 250M (5-10B tokens, 1× H200 SXM or 8× H100) | $200-500 | 1-2 days |
| Phase 3 | Base 1B at 1T tokens (8× B200 if available, else 8× H100/H200 SXM) | $11-25K | 9-30 days |
| Phase 3 caching | Teacher logit cache for distillation (3 teachers × 200B tokens × top-K=8) | $15-25K | 3-7 days (one-time) |
| **Phase 3 total (compute only)** | | **$26-50K** | **~2-5 weeks** |

If we cut to **500B tokens** for Phase 3 v1, halve the base cost: ~$13-25K total including cache. Discussed as an option.

---

## 9. The dossier audit's recent verdict (for context)

A subagent audit of the wind-tunnel debug session concluded:
- **GO** on running the sweep with single-source FineWeb-Edu (muP HP transfer is data-distribution-agnostic per Yang et al.)
- **RED for v1 release** if we ship without code, multilingual, or Nemotron-CC — this would be stack-validation, not a real release
- Recommended adding a **Section 6: Data Pipeline Preflight** to the dossier (which would have caught all the gating issues before pod launch)
- The 13-gram decontamination is unchanged from Llama 2 / OLMo 2 norms

---

## 9b. Debug investigation log — NaN at step 4200 (2026-05-11 PM)

Full record of how we narrowed down the NaN bug. Useful for you to assess whether the methodology was sound and the conclusion is right.

### Symptoms
First-cell training in the wind-tunnel sweep: loss curve looks healthy through step 4100 (~4.5-5.0 nats with ±0.4 step-to-step variance on FineWeb-Edu), then at **step 4200 exactly, loss = NaN**. Once NaN appears it persists (every subsequent step is NaN, including step 4300, 4400, ...).

### Isolation tests run (each ~25 min + $1.50 of pod time on H200 SXM)

| # | Variable changed | Other variables | Result | Implication |
|---|---|---|---|---|
| 1 | LR = 5e-4, init = 0.01 (initial cell) | seed=0, data_seed=1234, single-source FineWeb-Edu | NaN @ step 4200 | First observation |
| 2 | LR = 1e-3, init = 0.02 (Llama defaults) | seed=0, data_seed=1234, single-source FineWeb-Edu | NaN @ step 4200 | Not LR-sensitive at this range. Not init-sensitive. |
| 3 | LR = 1e-3, init = 0.02, **seed=42, data_seed=5678** | single-source FineWeb-Edu | NaN @ step 4200 | Different seeds didn't move it. BUT — with single-source data, data_seed has no effect (MixtureSampler has nothing to randomize between). So the data path was actually identical. |
| 4 | LR = 1e-4 (10× lower) | seed=42, single-source FineWeb-Edu | NaN @ step 4200 | **Not gradient-magnitude-driven.** If exploding gradients were the cause, NaN should've delayed proportionally to step ~42K. It didn't move. |
| 5 | **`--synthetic-data`** (random tokens, vocab_size=131072) | seed=0, total_steps=6000 | **No NaN.** Loss stayed at 11.797 ≈ ln(131072) (correct: random data has no learnable structure) | **Code path is innocent.** Forward+backward+optimizer work fine for 6000 steps on the same model. |

### Conclusion
The NaN is **data-driven**: a specific batch composition from FineWeb-Edu, deterministic by streaming order, produces a NaN forward pass at step ~4200. By step 4200 with batch=8 × seq_len=2048, we've consumed ~33,600 packed sequences = ~68 MB of tokens = ~138K FineWeb-Edu documents (at ~500 tokens/doc avg). Somewhere in that stream is a poisonous batch.

### Most likely failure mechanism
Gradient clipping (norm=1.0) didn't help → gradients are likely NaN *before* clipping runs, i.e. forward pass already produced NaN logits. Most plausible culprits:
1. A doc tokenizes to a sequence containing a rare Unicode token with near-zero (or zero) embedding norm; matmul through layer norm overflows
2. A doc with extreme repetition (200+ identical tokens) creates a degenerate attention pattern → softmax overflow
3. A doc with malformed UTF-8 produces an all-padding sequence after tokenization → all-zero embeddings → division by ~0 somewhere

We have NOT instrumented the loader to dump the actual batch contents at step 4200 yet — that's the next diagnostic step, but the NaN-skip patch unblocked the sweep so we deferred.

### Fix applied
`src/myllm/training/train_step.py` — added a standard NaN-skip block: detect non-finite loss or gradients, zero out the gradients on that step (effectively skipping the optimizer update), advance the step counter, continue. The optimizer's `m`/`v` moments stay clean because they accumulate squared-zero. Metrics now expose a `nan_skipped` counter so the loop logs how often this fires.

```python
# In train_step.py:
loss_finite = jnp.isfinite(loss)
grads_finite = jax.tree.reduce(
    lambda a, b: a & b,
    jax.tree.map(lambda g: jnp.all(jnp.isfinite(g)), grads),
    jnp.array(True),
)
step_ok = loss_finite & grads_finite

# Zero gradients on bad-batch steps so optimizer state stays clean
safe_grads = jax.tree.map(
    lambda g: jnp.where(step_ok, g, jnp.zeros_like(g)),
    grads,
)
updates, new_opt_state = optimizer.update(safe_grads, ...)
```

256 unit tests still pass. Tarball pushed to R2; sweep relaunching on the pod now.

### Why we're surfacing this debug log
Solo-led project — there's no peer who's seen this code or this dataset. **Anything you've seen on similar issues, even just "yes I've seen X, look at Y" is hugely valuable.** Questions 18-23 below dig deeper but the executive question is: **is NaN-skip in train_step alone the right primary defense, or are there other layers we should add (per-batch loss-outlier reject, FineWeb-Edu shard filtering, manifest pinning, etc.)?**

---

## 10. Questions for you (categorized)

### A. Architecture & training methodology
1. **muP width_mult range**: We sweep at width_mult=1.5 (67M proxy, hidden=384, base=256) and want transfer to width_mult=4 (pilot 250M) and width_mult=8 (base 1B). Some practitioners report transfer fidelity drops past 4× width_mult. Have you seen this? Should we sweep at a wider proxy (e.g., hidden=512, base=256 = width_mult=2) to be safer?

2. **QK-norm at 1B**: We're using QK-norm post-RoPE. Is this still net-positive at 1B scale, or only useful at ≥7B? Llama 3 doesn't use it at 7B+ but did at 1B/3B variants. Your call?

3. **GQA ratio**: We chose 3:1 (6 query heads, 2 KV heads at proxy; 16:4 at base 1B). Llama 3.2 1B uses 8:2 (4:1). Aggressive enough? Should we revisit?

4. **Tied embeddings**: Tied input/output. Should we untie for the base 1B to get independent lm_head gradients? Memory cost ~250MB. Worth it?

5. **WSD schedule fractions**: warmup=10%, stable=75%, decay=15%, end_lr=0.1×peak. Should the decay be longer (e.g., 25%)? End_lr lower (0.05× or 0.01×)?

6. **WSM merge count**: We plan to merge the last 3-5 stable-phase checkpoints. Literature shows up to 7 helps. Diminishing returns?

### B. Distillation
7. **3 teachers vs 5+**: We have DeepSeek-V4-Pro-Base + Qwen 3.6-27B + Mistral-Medium-3.5-128B. Some recent papers use 5-7 teachers. Diminishing returns past 3 for a 1B student?

8. **α=0.3 vs annealed α**: We use constant α=0.3 in the decay phase. Would annealed (e.g., α: 0.5 → 0.1 over decay) work better?

9. **Top-K=8 vs higher**: We cache top-K=8 teacher logits. Some work uses K=32. Trade-off is cache size (8GB-30GB) vs distillation fidelity. Worth it?

10. **Temperature=1.0**: We use T=1.0 for KL distillation (no soft-temperature). Should we go T=2 or T=4 for softer probabilities?

11. **Decay-phase activation at 85%**: We start distillation in the last 15% of training. Some papers start at 70% or 50%. Right fraction?

12. **Persona invisibility**: We use base (not chat) teachers + multi-teacher averaging + persona-strip SFT to make distillation invisible. Is this enough, or are there other tells (style markers, specific phrasings) we should worry about?

### C. Data
13. **1T tokens for 1B params (50× Chinchilla)**: Llama 3.2 1B used 9T (450×!), SmolLM2 1.7B used 11T (650×). Should we go bigger? Or is 1T enough for a v1?

14. **Math at 7%, code at 18%**: These shares are from a generic English-pretrain template. For an Indic-hedged model, should code be lower (e.g., 12%) and multilingual higher (e.g., 18%)?

15. **Hindi at 4%**: We have Sangraha at 4% of pretrain. Is this enough for "model can read/write Hindi"? Or do we need 10%+ for Indic capability?

16. **Decontamination 13-gram**: Standard for 1B+. Anyone using shorter (8-gram) at 1B scale for tighter filtering? Or is 13-gram sufficient?

17. **Tokenizer 131k vocab**: Big for a 1B model. Llama 3 used 128k. SmolLM2 uses 49k. Our 131k was chosen for Indic coverage (Devanagari needs lots of subword tokens). Right call?

### D. NaN-at-step-4200 — now narrowed to data path (see §9b for full investigation log)

18. **Is NaN-skip in `train_step` alone sufficient as the primary defense?** We applied the standard pattern: detect non-finite loss/grads → zero the gradients → optimizer effectively skips that step. Industry-standard but feels like a band-aid that hides the underlying data quality issue. Should we ALSO add per-batch loss-outlier rejection upstream (e.g., skip any batch whose loss is >3σ above running mean even when finite)? Or trust NaN-skip and move on?

19. **Have you seen specific FineWeb-Edu shards / docs cause NaN at this scale?** Some FineWeb-Edu shards are known to be flaky. The HF community has reported certain shards with malformed UTF-8 / extreme token repetition / placeholder content. Is there a known-bad-shard list we should filter out, or a more aggressive filter we should run before tokenization?

20. **High step-to-step variance**: Loss oscillates ±0.4 nats step-to-step on FineWeb-Edu (vs <0.1 in Llama 3 reports). Is this normal for a 67M-on-single-source mix, or a signal of something deeper? Once we have the production multi-source mix back, will the variance drop because batches average across sources?

21. **How would you find the specific poisonous doc?** Pragmatic approach for instrumented loading? Options we considered: (a) hook in `HFStreamLoader` to log doc_id + first 200 chars at every 100th doc, then post-mortem; (b) checkpoint at step 4100, re-run + inspect batch at step 4200 with `jax.debug.print`; (c) just compute embedding-norm distribution over the docs and see if any are pathological. Which would you reach for first?

### E. Compute & infrastructure
22. **B200 vs H200 SXM for Phase 3**: B200 is fastest ($5.49/hr, ~150K tok/sec/GPU on 1B) but currently "Unavailable" on RunPod secure cloud. H200 SXM ($3.99/hr) is what we're on now. For a 10-30 day run, is spot/preempt risk worth saving 20-30%? Or pay for reserved capacity?

23. **Multi-node FSDP**: Our `myllm.training.mesh` supports single-node 8× FSDP. Should we plan multi-node for Phase 3 (more memory, longer runs) or single-node is fine at 1B?

24. **Checkpoint cadence**: We checkpoint every 1000 steps (Phase 2/3) → ~10 GB each. At 1T tokens / 250K steps total, that's 250 checkpoints. R2 storage is fine ($15 / TB-month). Sane?

### F. Eval & alignment
25. **Eval coverage**: We have MMLU-ProX (29 langs) + Belebele (122 langs) + MILU (Indic). Missing: HumanEval+, MBPP+, GSM8K, IFEval (instruction-following), MT-Bench (chat). Which of these would you add for a 1B base?

26. **Held-out validation**: We don't currently have a held-out validation split for monitoring train→eval loss gap. Just last-batch loss. Worth adding?

27. **CoT prompting at 1B**: Most 1B models can't reliably CoT. We're using direct-answer prompts (single-letter A-J). Right call, or should we add a CoT extractor for evals?

### G. Project-level / strategic
28. **1B vs 3B**: We're locked at 1B. If you started today, would you go 3B instead? Cost is ~4× but the model is meaningfully more usable.

29. **Release scope without Nemotron-CC**: If NVIDIA approval takes 3+ months, do we (a) wait, (b) substitute with DCLM-baseline (ungated, similar quality), or (c) ship and add Nemotron-CC in v2?

30. **Solo lead risk**: I'm the only person on this. What's the most likely failure mode you've seen on similar solo-led pretrain projects, and how should I mitigate it?

31. **Was there a simpler path?**: Reading the above architecture, are we over-engineering for a 1B model? Should we have just done a Llama 3.2 1B clone + scaling, without muP/WSM/distillation, and shipped faster?

32. **Most important thing I'm missing**: From this brief, what's the biggest thing you'd flag as "wait, why are you doing X?" Or "you should also do Y."

---

## 11. What we'd love from you, specifically

In order of value:

1. **30 minutes of your time on Questions 18-21** (the NaN debug + how to find the bad doc). See §9b for the investigation log. Even just "yes I've seen that, look at X" is huge.

2. **A quick sanity check on the muP setup** (Questions 1-4). We've tested it in unit tests but haven't validated the transfer empirically at scale yet.

3. **Your read on the data mix and gated-data contingency** (Questions 13-17, 29). This is where I have the least confidence.

4. **Anything from Section G you can answer in <5 sentences** (Questions 30-32). Honest external take on the project shape.

If you have a couple of hours to look deeper, I'd value (in order):
- A code review of `src/myllm/training/loop.py` + `src/myllm/training/optimizer.py` (the muP + WSD + watchdog interaction is the most subtle code in the repo)
- A review of `configs/base_1b.yaml` against your default reference 1B architecture
- A look at our eval harness's MMLU-ProX adapter (`src/myllm/eval/benchmarks/mmlu_prox.py`) — we extract single-letter answers with prefix-stripping regex; want to make sure we're not over-crediting or under-crediting models

---

## 12. Repo access

The repo is now on git (initial commit `6834eb9` on `main`, ~17.5K lines). I'll push to a private GitHub repo and send you the URL — happy to add you as a collaborator with whatever permission you prefer (read-only / triage / write).

Code tarballs are also pushed to R2 every major change (current: `s3://llm-data/code/llm-build-20260511-203228.tar.gz` SHA `6e7997e6f30cf40e55717dc61ec7dae99e5c3ea60655c3611b8205d51cee1155` — includes the NaN-skip patch).

Tests: `pytest --no-header -q` (256 pass, 4 pre-existing skips for keras+TF).
Lint: `ruff check src/ tests/ scripts/`.

**Files I'd suggest you look at first** (in order):
1. This doc + §9b for the NaN debug investigation
2. `src/myllm/training/train_step.py` — the NaN-skip patch + the muP-aware update + (optional) distillation switch
3. `src/myllm/training/optimizer.py` — muP per-param LR scaling via `optax.multi_transform`
4. `configs/wind_tunnel.yaml` — 67M proxy architecture (sweep target)
5. `configs/base_1b.yaml` — Phase 3 target architecture
6. `src/myllm/training/loop.py` — full training loop (watchdog, checkpoint, resume, decay-phase hook)
7. `src/myllm/data/loader.py` + `pretrain_mix.yaml` — data pipeline state

---

## 13. Closing note

This is a real "0 to model" build, not a "fine-tune Llama" project. The core risk is **death by a thousand small wrongnesses** rather than any single big mistake — and that's exactly the failure mode a senior outside reviewer catches and I don't. Anything you spot is appreciated.

Thanks for taking the time.

— harshit.hv@samatva.com
