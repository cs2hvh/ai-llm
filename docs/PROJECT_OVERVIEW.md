# MyLLM — Project Overview

**Status doc, kept current.** Single source of truth for "what is this project, how does it work, where is it." If something in here is out of date, fix it first.

**Last update:** 2026-05-13 (evening) — FSDP shipped + validated on 2× H200 SXM; dual-mode decontam wired; code-only source built (180M tokens); audit machinery green; **Stage 1 pilot unblocked**.
**Lead:** harshit.hv@samatva.com (solo, treats project as enterprise)
**Build partner:** Claude (Anthropic) — pair-programming + research agent fleet
**Repo:** https://github.com/cs2hvh/ai-llm — `main` branch is canonical

---

## 1. What is MyLLM?

A **1B-parameter decoder-only foundation model**, trained from scratch with a multi-teacher distillation phase, on a 12-source multilingual corpus (English-primary + Hindi/Indic hedge + code + math + Q&A). The goal: a defensible v1 small LLM that demonstrates an end-to-end enterprise-grade pretraining stack on commodity GPU pods (5-8× H200 SXM scale).

**Why this project exists:**
- Prove the full pretraining stack (data → tokenizer → pretraining → distillation → eval gates → release) can be built and run by a small team
- Hedge English-only foundation models with explicit Hindi/Indic representation
- Distill from large teachers (DeepSeek-V4-Pro + Olmo-3-32B) into a deployable 1B size

**Why 1B specifically:** sub-1B is too small to demonstrate the recipe at scale; 7B+ is operationally out of reach without large clusters. 1B is the sweet spot for showing enterprise rigor while staying buildable.

---

## 2. Project status — high-level scoreboard

```
┌──────────────────────────────────────────────────────────────────┐
│                    CURRENT STATE (2026-05-13)                    │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│   PHASE                              STATUS                      │
│   ─────────────────────────────────  ──────────────────────      │
│   Phase 1: Tokenizer + design        ✓ DONE (R2-hosted)          │
│   Phase A: Bug fixes + audit         ✓ DONE                      │
│   Phase B: Corpus pipeline           ✓ DONE (820M staged)        │
│   Canary ladder (L0/L1/L3/L5)        ✓ ALL PASS on 5×H200        │
│   Throughput baseline                ✓ Measured (1B+250M)        │
│   Senior review round 1              ✓ Processed                 │
│   Senior review round 2              ✓ All P0 fixes shipped      │
│   FSDP-in-JAX (Commits A–G)          ✓ DONE (gauntlet G1-G4 PASS │
│                                        on 2× H200 SXM 2026-05-13)│
│   Code-only source (codeparrot)      ✓ DONE (180M tokens, R2)    │
│   Dual-mode decontam (8+13 gram)     ✓ DONE (wired+indexed, 10   │
│                                        benchmarks on R2)         │
│   Teacher audit machinery            ✓ DONE (real-text rerun TBD)│
│                                                                  │
│   ▶ Compose v2 1B corpus             ◐ NEXT (~30 min on GPU pod) │
│   ▶ Re-bench 1B-shape post-FSDP      ◯ during Stage 1 warmup     │
│   ▶ Stage 1: 250M pilot @ 30-50B     ◐ READY TO LAUNCH           │
│   ▶ Stage 2: 1B rehearsal @ 10-30B   ◯ blocks on Stage 1 green   │
│   ▶ Stage 3: Base v1 (600B tokens)   ◯ blocks on Stage 2 green   │
│   ▶ Release scorecard                ◯ blocks on base run        │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

**Reference today's commits** (chronological): cc56daa wires dual-mode decontam through `build_packed_corpus.py`; aaff2fb fixes the mmlu-prox column-schema bug; b1dd5dd ships the teacher top-K mass audit script + 14 tests; eab98c9 lands `pod_setup_apt.sh` + `pod_launch_gpu.sh` + `run_fsdp_gauntlet.sh` + `run_teacher_audit.sh`; 7002d50 isolates gauntlet checkpoints from real-run state; 6cf299e pins the canonical `torch==2.7.1 + jax[cuda12]==0.4.38` single-venv recipe after a 7-attempt CUDA-version saga.

---

## 3. Tech stack

| Layer | Choice | Why |
|---|---|---|
| **Backend framework** | Keras 3 (JAX backend) | Pythonic, multi-backend, native sharding via `jax.sharding`. Avoids being locked into either PyTorch or pure Flax. |
| **Compute graph / autodiff** | JAX 0.4.x | XLA compilation, native multi-device, deterministic re-runs |
| **Optimizer** | Optax (`adamw` + `multi_transform` for muP groups) | Composable, fp32-moments-pinned (P0-2 fix), supports per-group LR scaling |
| **Tokenizer** | **SentencePiece-Unigram**, 131,075 runtime vocab, byte fallback | Trained natively via SentencePiece (HF tokenizers' Python BPE hits a ~15GB corpus ceiling). Wide vocab supports Indic + code without exploding fragmentation. |
| **Data storage** | **Cloudflare R2** (S3-compatible) | Cheap egress, fast (32-way parallel multipart upload → 954 Mbps measured). Per-source corpora + composed mixed corpus + tokenizer + checkpoints all live there. |
| **Data pipeline** | HuggingFace `datasets` streaming | One stream per source, filter chain → tokenize → pack → write to R2 |
| **Checkpoint** | **Orbax** (sharded, templated restore) | Atomic shard writes, per-step manifest.json for partial-write detection, templated restore preserves namedtuples (muP `MultiTransformState`) |
| **Sharding (current)** | DP-replicated state via `jax.sharding.NamedSharding(mesh, P("data"))` for inputs | **Limitation**: optimizer state is replicated, not sharded. Hits OOM at 5-GPU 1B/seq=8192. |
| **Sharding (planned)** | **FSDP/ZeRO-3** via `NamedSharding` on params + opt state | Lets us fit 1B/seq=8192 at mb≥2 per device. ~3-5 dev-days. |
| **Logging** | structlog (JSON events) + W&B for training runs | Greppable, machine-parseable; W&B for human dashboarding |
| **Tests** | pytest, ~457 tests across 39 files | All green; no GPU dependency for the core suite |

**Why JAX over PyTorch**: deterministic JIT, native multi-device via shardings (no separate FSDP library), cleaner functional patterns. The cost is a smaller ecosystem and harder ZeRO-3 implementation than PyTorch+FSDP. Trading ecosystem for control.

**Why Keras 3 wrapper on top of JAX**: layer composition + Variable lifecycle without writing pure Flax. Lets us reuse community attention / norm implementations.

---

## 4. System architecture (high-level)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            MyLLM PRETRAINING SYSTEM                          │
└─────────────────────────────────────────────────────────────────────────────┘

                                   ┌────────────────────────┐
                                   │   Cloudflare R2 (CDN)  │
                                   │  - source corpora      │
                                   │  - composed corpus     │
                                   │  - tokenizer (v2)      │
                                   │  - checkpoints         │
                                   │  - teacher logit cache │
                                   └───────────▲────────────┘
                                               │
                                               │ multipart parallel I/O
              ┌────────────────────────────────┼────────────────────────────────┐
              │                                │                                │
              │                                │                                │
    ┌─────────▼─────────┐         ┌────────────▼──────────┐         ┌──────────▼────────┐
    │ DATA CONSTRUCTION │         │   TRAINING SYSTEM      │         │  EVAL + RELEASE   │
    │  (CPU server)     │         │   (GPU pod 5-8× H200)  │         │  (CPU + GPU)      │
    │                   │         │                        │         │                   │
    │  HF datasets ──┐  │         │  PackedCorpusReader    │         │  decontam_index   │
    │  filter chain ─┤  │         │       │                │         │       │           │
    │  tokenize  ────┤  │         │       ▼                │         │       ▼           │
    │  pack    ──────┤  │         │  iter_packed_pairs     │         │  L5 verify        │
    │  per-source ───┘  │         │       │                │         │  MMLU, GSM8K,     │
    │       │           │         │       ▼                │         │  HumanEval, etc.  │
    │       ▼           │         │  train_step (JIT)      │         │       │           │
    │   compose mix     │         │  ┌─────────────────┐  │         │       ▼           │
    │       │           │         │  │ model.forward   │  │         │  release scorecard│
    │       ▼           │         │  │   ↓ hidden      │  │         │  + memorization   │
    │   L5 verify       │         │  │ chunked CE loss │  │         │    probes         │
    │       │           │         │  │   ↓ grads       │  │         │                   │
    └───────┼───────────┘         │  │ optimizer.update│  │         └───────────────────┘
            │                     │  │   ↓ new state   │  │
            │ upload              │  │ Orbax ckpt save │  │
            ▼                     │  └─────────────────┘  │
        R2 corpus_v1/             │       │                │
                                  │       ▼                │
                                  │  R2 ckpt mirror        │
                                  └────────────────────────┘
```

**Key design choices visible in this diagram:**
- **R2 is the central nervous system.** All durable artifacts go through R2. The CPU server and GPU pod never directly share filesystems — they communicate via R2.
- **Data is pre-tokenized into packed sequences offline.** Training reads from `PackedCorpusReader`, not from raw HF streams. This is the **B2 design**: pretokenization removes a major variability source during training.
- **Eval runs separately.** Checkpoints get pulled from R2, eval runs on a separate pod (or CPU for cheap checks).

---

## 5. Data pipeline — detailed flow

```
PER-SOURCE BUILD (scripts/build_packed_corpus.py)
══════════════════════════════════════════════════
                              ┌──────────────────────────────┐
                              │  configs/data/pretrain_mix   │
                              │  - dataset name + share      │
                              │  - text_field, config, split │
                              └──────────────┬───────────────┘
                                             │
                          ┌──────────────────▼──────────────────┐
                          │   HFStreamLoader (one per source)   │
                          │   `datasets.load_dataset(streaming) │
                          └──────────────────┬──────────────────┘
                                             │ Documents
                          ┌──────────────────▼──────────────────┐
                          │   FilterChain                       │
                          │   - length (200 ≤ chars ≤ 1M)       │
                          │   - repetition (top-word + 5gram)   │
                          │   - symbol ratio (≤ 30%)            │
                          │   - PII (email/phone redact)        │
                          │   - decontam* (if index present)    │
                          └──────────────────┬──────────────────┘
                                             │ KeptDocuments
                          ┌──────────────────▼──────────────────┐
                          │   Char-aware adaptive batch flush   │
                          │   (Phase A fix: prevents over-fetch)│
                          └──────────────────┬──────────────────┘
                                             │ DocBatches
                          ┌──────────────────▼──────────────────┐
                          │   HF Tokenizer (Rayon batch encode) │
                          │   ~14× speedup over per-doc encode  │
                          └──────────────────┬──────────────────┘
                                             │ list[int] (token ids)
                          ┌──────────────────▼──────────────────┐
                          │   SequencePacker (seq=context+1)    │
                          │   - EOS-separated multi-doc packs   │
                          │   - segment_ids for intra-doc mask  │
                          └──────────────────┬──────────────────┘
                                             │ PackedSequences
                          ┌──────────────────▼──────────────────┐
                          │   MinHash+LSH dedupe                │
                          │   (112 perms, 14 bands, J=0.75)     │
                          └──────────────────┬──────────────────┘
                                             │
                          ┌──────────────────▼──────────────────┐
                          │   PackedCorpusWriter                │
                          │   - tokens.bin (uint32, on disk)    │
                          │   - seq_meta.arrow                  │
                          │   - doc_meta.parquet                │
                          │   - manifest.json (per-shard)       │
                          └──────────────────┬──────────────────┘
                                             │
                          ┌──────────────────▼──────────────────┐
                          │   R2 streaming mirror (32-way par.) │
                          │   then delete-local-after-upload    │
                          └─────────────────────────────────────┘


COMPOSE MIXED CORPUS (scripts/compose_mixed_corpus.py)
══════════════════════════════════════════════════════
        per-source corpora on R2 (12 sources)
                       │
                       ▼ download (~30s parallel)
        ┌──────────────────────────────────────────┐
        │  Open N PackedCorpusReaders              │
        │  Validate: tokenizer_sha + seq_len agree │
        └────────────────┬─────────────────────────┘
                         │
                         ▼ deficit-driven sampling
        ┌──────────────────────────────────────────┐
        │  Loop:                                   │
        │    pick_source() based on target share   │
        │    read 1 packed sequence                │
        │    remap doc_span ids                    │
        │    writer.append_sequence(tokens, spans) │
        │  until all sources exhausted             │
        └────────────────┬─────────────────────────┘
                         │
                         ▼
        ┌──────────────────────────────────────────┐
        │  write_corpus_manifest                   │
        │  Aggregate: total_seq, total_tokens,     │
        │  actual_source_share                     │
        └────────────────┬─────────────────────────┘
                         │
                         ▼
        ┌──────────────────────────────────────────┐
        │  L5 verify:                              │
        │   ✓ manifest complete (all shards)       │
        │   ✓ tokenizer_sha uniform                │
        │   ✓ source-share drift ≤ 2%              │
        │   ✓ token range [0, vocab)               │
        │   ✓ segment_ids well-formed              │
        └────────────────┬─────────────────────────┘
                         │
                         ▼ upload to R2/corpus_v1/train
        ┌──────────────────────────────────────────┐
        │  Training-ready packed corpus on R2      │
        └──────────────────────────────────────────┘
```

**Current state (as of 2026-05-12 evening):**
- 12-source `infra-validation corpus v1` built → 808.5M tokens on R2 at `s3://llm-data/corpus_v1/train/`
- **NOTE**: built at seq=8192 due to a default-value bug (P0-3 from reviewer round 2); needs rebuild at seq=4097 (for pilot/4k base) before any training. Fix landed in commit `d66899e`.
- Sources skipped: starcoderdata (gated), proof-pile-2 (loader errors), the-stack-v2 (content out-of-band)

---

## 6. Model architecture

**Llama-style decoder-only**, with 2024-2025 standard recipe upgrades. Source: [src/myllm/model/transformer.py](src/myllm/model/transformer.py), [src/myllm/model/layers.py](src/myllm/model/layers.py).

```
TransformerLM (1B)
══════════════════
   input_ids  [B, S]
       │
       ▼
   ┌─────────────────────┐
   │ Token Embedding     │  [V, H] = [131075, 2048] → 0.27B params (40% of total)
   │ tied with LM head   │
   └──────┬──────────────┘
          │ x: [B, S, H]
          ▼
   ┌─────────────────────┐    ┌──────────────────────────────────┐
   │  DecoderBlock × 16  │ ──→│  Each block:                     │
   │  (jax.checkpoint ON)│    │   x  ─→ RMSNorm ─→ Attention ─┐ │
   └──────┬──────────────┘    │                                │ │
          │                   │   GroupedQueryAttention:       │ │
          │                   │     - 32 query heads           │ │
          │                   │     - 8 KV heads (GQA 4:1)     │ │
          │                   │     - head_dim = 64            │ │
          │                   │     - RoPE (base=130000)       │ │
          │                   │     - QK-norm (RMSNorm/head)   │ │
          │                   │     - jax.nn.dot_product_atn   │ │
          │                   │       (cuDNN flash on H100+)   │ │
          │                   │     - segment_ids causal mask  │ │
          │                   │                          ◀─────┘ │
          │                   │       residual add               │
          │                   │   x  ─→ RMSNorm ─→ FFN  ─┐       │
          │                   │                          │       │
          │                   │   SwiGLU FFN:            │       │
          │                   │     gate_proj [H, F=8192]│       │
          │                   │     up_proj   [H, F]     │       │
          │                   │     SiLU(gate) * up      │       │
          │                   │     down_proj [F, H]     │       │
          │                   │                     ◀────┘       │
          │                   │       residual add               │
          │                   └──────────────────────────────────┘
          │                                
          ▼ final hidden: [B, S, H]
   ┌─────────────────────┐
   │ Final RMSNorm       │
   └──────┬──────────────┘
          │
          ▼ TWO PATHS (set via return_loss_inputs flag):
          │
   ┌──────┴────────────────┐         ┌────────────────────────────────┐
   │ Path A: full logits    │        │ Path B: loss inputs            │
   │ (default; small vocab) │        │ (chunked CE; large vocab)      │
   │   logits = x @ E.T     │        │  return (hidden, E, mult)      │
   │   [B, S, V] ~ 8.6 GB   │        │  loss function streams vocab   │
   │   at production batch  │        │  in chunks; never materializes │
   │                        │        │  [B, S, V]                     │
   └────────────────────────┘        └────────────────────────────────┘
```

**Architectural decisions and rationale:**

| Decision | Setting | Rationale |
|---|---|---|
| Attention type | GQA 4:1 (32 query / 8 KV) | Llama-3 ratio; cuts KV-cache 4× without loss-quality regression |
| Position encoding | RoPE base=130000 | Long-context friendly (Yarn-style); same as Llama-3 |
| Normalization | RMSNorm (pre-norm, eps=1e-5) | Industry standard; QK-norm added 2026-05-11 (stability) |
| Activation | SwiGLU, FFN=4× hidden | Llama-3 standard; +ffn_dim=8192 |
| Embedding tying | True | Saves ~270M params (~22% of total at 1B) |
| muP | base_width=256, all 3 multipliers on | Zero-shot HP transfer from wind-tunnel (h=384) to base (h=2048) |
| z-loss | coef=1e-4 | Stabilizes long pretrain (PaLM, OLMo recipe) |
| QK-norm | True | muP HP-transfer requires consistent QK-norm across chain |
| scaled_init_for_residuals | True | Matches base; required for transfer validity |
| Gradient checkpointing | True (per DecoderBlock) | Required to fit 1B/seq=8192 on H200; ~33% recompute tax, ~5× activation-memory win |

---

## 7. Training loop architecture

```
TRAINING STEP (JIT-compiled, JAX functional)
═══════════════════════════════════════════════════════════════════

state = {
    "trainable_variables":     [params],
    "non_trainable_variables": [RoPE tables, ...],
    "opt_state":               AdamW state (m, v) + multi_transform groups,
    "step":                    int32,
    "lr_recovery_multiplier":  float32,   # halved on watchdog spike
    "data_position":           int32      # cumulative tokens consumed
}

def train_step(state, batch):
    # 1. FORWARD + LOSS
    if use_chunked_ce and no_teacher:
        # NEW path (avoids [B,S,V] materialization)
        hidden, lm_head_w, mult = model.stateless_call(
            trainable, non_trainable, batch["input_ids"],
            return_loss_inputs=True
        )
        loss, metrics = chunked_cross_entropy_with_z_loss(
            hidden, lm_head_w, batch["labels"],
            num_chunks=8, output_mult=mult, ignore_index=pad_id
        )
    else:
        logits, _ = model.stateless_call(...)
        loss, metrics = distillation_mixed_loss(logits, batch["labels"], ...)

    # 2. BACKWARD
    grads = jax.value_and_grad(loss_fn)(...)

    # 3. ATOMIC NaN-revert
    is_finite = jnp.all(jnp.isfinite(grads))
    # candidate update
    updates, new_opt_state = optimizer.update(grads, state["opt_state"], ...)
    candidate_trainable = optax.apply_updates(state["trainable_variables"], updates)
    # take EITHER candidate or old state via jnp.where — atomic
    new_trainable = jnp.where(is_finite, candidate, old)
    new_opt_state = jnp.where(is_finite, new_opt_state, state["opt_state"])

    # 4. SHARDING (planned, FSDP)
    new_trainable = with_sharding_constraint(new_trainable, param_sharding)

    return new_state, metrics
```

**Loop wrapper (`src/myllm/training/loop.py`):**
```
for batch in data_iter:
    if step >= total_steps: break
    
    # decay-phase distillation injection (last 15%)
    if decay_phase.is_active(step):
        batch = decay_phase.maybe_inject(state, batch)  # adds teacher_topk
    
    state, metrics = train_step(state, batch)
    state["data_position"] += B * input_ids.shape[1]
    
    # NaN-detect: if revert happened, log + write to quarantine
    if metrics["nan_skipped"] > 0:
        quarantine.write(batch, reason="nan_skipped")
        continue
    
    # Loss-spike watchdog (rollback + LR multiplier × 0.5 on hard spike)
    if watchdog.observe(loss) == "hard":
        state = recover_from_spike(state, ckpt)  # restore from prev ckpt
        consumed_iter.skip(K)  # skip K batches past spike
    
    # Checkpoint (cadence: every 25B tokens for production)
    if step % checkpoint_every == 0:
        ckpt.save(step, state_to_save, extra={"data_position": ...})
```

---

## 8. Canary ladder — what's verified

The project uses a **5-tier canary ladder** to validate training infrastructure before committing to long runs. Each tier catches a specific class of bug.

```
L0 — Static checks (CPU, instant)
  ├─ Tokenizer roundtrip (208 strings)
  ├─ Model config self-consistency (heads × head_dim = hidden)
  └─ Status: ✓ ALL PASS

L1 — Single-GPU smoke (1 GPU, 2-5 min)
  ├─ 20 steps synthetic data
  ├─ Loss finite, near-random-init (~log(V))
  ├─ Checkpoint save (Orbax → R2 mirror)
  └─ Status: ✓ PASSED on 5×H200 pod

L3-synthetic — Forced-kill resume bitwise-exact (CPU, 30-60s)
  ├─ Run N steps uninterrupted → final state hash
  ├─ Run N/2 steps, save ckpt, resume → run to N → final state hash
  ├─ Assert: hashes match
  ├─ Caught: synthetic-iter non-resume-safe (fixed via start_step param)
  └─ Status: ✓ PASSED

L3-packed — Same as above BUT real packed-corpus reader path (CPU, ~60s)
  ├─ Tiny tokenizer + 30-sequence packed corpus on disk
  ├─ Tests production data path (seek, segment_ids, data_position)
  ├─ Caught: off-by-one in seek (packed_seq_len vs context_length)
  └─ Status: ✓ PASSED (after fix)  ← CAUGHT A REAL PROD BUG

L5 — Source-share drift on composed corpus (CPU, ~30s)
  ├─ Verify all shards have manifests
  ├─ Tokenizer SHA uniform across shards
  ├─ Source-share drift ≤ 2% (vs target from pretrain_mix.yaml)
  ├─ Token range [0, vocab)
  ├─ Segment_ids well-formed
  └─ Status: ✓ 5/5 PASSED on 808M corpus

   ▶ L2 — Multi-GPU loss parity (PLANNED, ~5 min on FSDP-ready pod)
     ├─ Run N steps single-device with seed=42
     ├─ Run N steps sharded across N devices with seed=42
     ├─ Assert: loss curves agree within 5e-3 per step
     └─ Status: ◯ NOT YET BUILT (Commit E of FSDP plan)

   ▶ L3-multigpu — Resume bitwise-exact under FSDP (PLANNED)
     ├─ Same as L3-packed but with sharded state
     └─ Status: ◯ NOT YET BUILT
```

**Why this ladder works:** each tier catches a specific bug class. L3-packed in particular surfaced a silent corruption bug that would have ruined every production resume — exactly what the canary is for.

---

## 9. Throughput + cost model (measured + projected)

**Measured on 5×H200 SXM (single pod, current DP-replicated state):**

| Config | tok/sec/device | MFU | Peak HBM | 1T extrap (5×H200, linear) |
|---|---|---|---|---|
| 250M @ seq=4096, mb=4 | 47K | 3.4% | 104 GB | ~138 days |
| 1B @ seq=4096, mb=1 | 15.1K | 5.7% | 91 GB | ~154 days |
| 1B @ seq=8192, mb=1, grad-ckpt | 8.7K | 3.3% | 61 GB | ~267 days |
| 1B @ seq=8192, mb=5 / 5GPU | OOM | — | 140 GB/dev | n/a |

**Cost model (RunPod ~$3.5/hr/GPU):**

For our locked **600B v1 target**:

| Path | Throughput | Wall (5×H200) | Cost |
|---|---|---|---|
| Pre-FSDP, seq=4096 | 75K agg | ~92 days | ~$39K |
| **Post-FSDP, seq=4096, mb≥4** | ~225K agg est | **~30 days** | **~$13K** ← preferred |
| Post-FSDP, seq=8192, mb≥2 | ~150K agg est | ~50 days | ~$21K |

**Industry MFU references (for context):**
- TinyLlama 1.1B: 56% MFU (A100, seq=2048, FSDP, no grad-ckpt)
- OLMo 2 32B: 38% MFU (H100, seq=4096, FSDP)
- hackbot.dad 1B Llama-3.2: 40% MFU (8×H100, seq=4096, DDP)
- **Our current: 3-7% MFU** — gap is structural (replicated state + grad-ckpt + 131k vocab + seq=8192)

The reviewer's 18-20% MFU target requires FSDP + removing grad-ckpt + larger batch. Reachable but requires the FSDP work.

---

## 10. Staged training plan

Adaptive plan based on the senior reviewer's recommendation. Each stage validates the next.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        STAGED TRAINING PLAN                              │
└─────────────────────────────────────────────────────────────────────────┘

STAGE 0 — FSDP + canary validation (no real training)
  Duration:    3-5 dev-days code + 0.5 day GPU validation
  Goal:        FSDP working + L2/L3 canaries green
  Compute:     ~$200-500 on a brief GPU session
  Gate:        L2 parity within 5e-3, L3-multigpu bitwise-exact
  Status:      ◐ STARTING TOMORROW

STAGE 1 — Pilot (250M @ 30-50B tokens)
  Duration:    2-3 days wall
  Goal:        End-to-end pipeline validation (corpus → train → eval)
  Compute:     ~$1K
  Gate:        Loss curve smooth, watchdog quiet, no NaN, eval suite runs
  Status:      ◯ blocks on Stage 0

STAGE 2 — 1B systems rehearsal (10-30B tokens)
  Duration:    3-5 days wall
  Goal:        Validate the 1B-shape model trains stable + at expected MFU
  Compute:     ~$2-3K
  Gate:        Throughput ≥ 80% of expected; eval curves moving
  Status:      ◯ blocks on Stage 1

STAGE 3 — Base v1 (600B tokens @ 1B params)         ◀── THE REAL RUN
  Duration:    ~30 days wall (FSDP, seq=4096) or ~50 days (seq=8192)
  Goal:        Ship v1 base model
  Compute:     ~$13-21K
  Gate at 300B (floor):  do not ship below this
  Gate at 400B:          if eval gain < 1.5%/50B × 2 intervals → STOP
  Gate at 500B:          eval cadence + decide ramp toward 1T
  Gate at 600B (target): if last 100B gain ≥ 3% → continue toward 1T
                         else → STOP, declare v1
  Hard ceiling:          1T tokens
  Status:      ◯ blocks on Stage 2

STAGE 4 — Continue toward 1T (only if curves justify)
  Conditional. Triggered by the Stage 3 gate.

POST-PRETRAINING (not in scope of this doc):
  - SFT (supervised fine-tuning) on instruction data
  - DPO or similar preference alignment
  - Long-context extension via YaRN (4K → 8K → 32K)
  - Quantization for deployment (int8, int4)
  - Release scorecard + model card
```

**Adaptive stop rule (for Stage 3):**

```
   eval_score
       │
   high│                                  ╭──── continue toward 1T
       │                            ╭─────╯     (gain ≥ 3% per 100B)
       │                       ╭────╯
       │                  ╭────╯
       │             ╭────╯           ◀── 600B target (re-decide here)
       │        ╭────╯
       │     ╭──╯
       │   ╭─╯
       │  ╱
       │ ╱   ◀── 300B floor (don't ship under)
       │╱
   low │ 
       └────┬─────┬─────┬─────┬─────┬─────┬────  tokens
           0    100B  200B  300B  400B  500B  600B  700B...
                              │
                              └── early-stop window (if gain < 1.5%/50B × 2)
```

---

## 11. Distillation strategy

```
DECAY-PHASE MULTI-TEACHER DISTILLATION
═══════════════════════════════════════════════════════

Phase 1: Stable training (first 85% of tokens)
  - Pure cross-entropy on next-token prediction
  - No teacher in batch
  - α = 1.0 (CE weight)

Phase 2: Decay (last 15% of tokens)
  - Teachers: DeepSeek-V4-Pro-Base + Olmo-3-32B-Base
  - Top-K = 64 logits per teacher per token (offline cached)
  - Loss = α × CE + (1-α) × KL(teacher_avg || student)
  - α anneals 0.7 → 0.3 across decay window
  - Cache lookup by data_position (corpus-aligned)

TEACHER LOGIT CACHE STORAGE
   tokens passed through teacher offline
        │
        ▼
   for each token, save: (top_64_logits, top_64_indices)
        │
        ▼
   uint16 indices + bf16 logits → ~256 bytes per token
        │
        ▼
   600B tokens × 256 bytes = ~150 TB of teacher cache
        │
        ▼
   Pragmatic: only cache for the decay phase (15%) = ~22 TB
   Stored on R2; streamed during decay phase only

WHY DECAY-PHASE ONLY:
  - Compute saving: 85% of training doesn't pay teacher-inference cost
  - Quality: distillation is most useful late (student has learned basics)
  - Storage: 22 TB cache is doable on R2; 150 TB would not be
  - Honest gain: ~1.15-1.20× effective tokens (NOT 1.5-3× of throughout)
```

---

## 12. Risk register (what could still go wrong)

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| FSDP implementation has subtle parity bug | Medium | Catastrophic (silent training corruption over weeks) | L2 canary, HLO-grep assertion, gauntlet G1-G6 |
| Loss-spike watchdog fires spuriously | Low | Days lost to false rollback | Have not stress-tested in real training; pilot rehearsal needed |
| Distillation teacher cache infrastructure bug | Medium | Silent quality degradation in decay phase | Top-K mass audit + teacher-cache canary before commit |
| Multi-GPU NCCL hang | Medium | Run stalls; need restart | Watchdog timer + checkpoint resume |
| Pod evict mid-run | High (RunPod is spot-ish) | 0-50 GB lost work between checkpoints | 25B-token checkpoint cadence; R2 mirror; resume tested via L3-packed |
| Eval pipeline data leak | Medium | Inflated benchmark scores; reputational risk | Decontam dual-mode (8gram + 13gram); fail-closed in --production |
| muP transfer doesn't actually transfer | Medium | Wrong HP at base scale; loss diverges | Proxy A → Proxy B → base ladder validates transfer empirically |
| Corpus quality is too low | Medium | Lower-than-projected eval scores | Per-source val splits + memorization probes; staged eval cadence |
| Hardware sourcing fails | Low | Project on hold | RunPod + Lambda fallbacks identified |

---

## 13. Where things live (file map)

```
llm-build/
|-- configs/
|   |-- base_1b.yaml                 <- pre-1 1B model spec and training target
|   |-- pilot_250m.yaml              <- 250M pilot config
|   |-- wind_tunnel.yaml             <- muP HP-sweep config
|   |-- pre2_dense_1_5b.yaml         <- pre-2 dense target planning config
|   |-- pre2_dense_proxy_400m.yaml   <- pre-2 proxy-study planning config
|   |-- pre2_moe_1b_active_research.yaml <- MoE research-only planning config
|   `-- data/
|       |-- pre2_mix_stage1.yaml     <- pre-2 foundation mix planning config
|       |-- pre2_mix_stage2.yaml     <- pre-2 capability mix planning config
|       |-- pre2_mix_anneal.yaml     <- pre-2 anneal/context mix planning config
|       `-- pretrain_mix.yaml        <- pre-1 pretrain mix, decontam, filter chain
|-- docs/
|   |-- PROJECT_OVERVIEW.md          <- this file
|   |-- PRE2_FINAL_PLAN_2026-05-14.md <- current pre-2 architecture, data, training, and hardware plan
|   |-- PRE2_ARCHITECTURE_DECISION.md <- accepted pre-2 architecture decision
|   |-- PRE2_STACK_MIGRATION_PLAN.md <- PyTorch/TorchTitan migration backlog
|   |-- mup_design.md                <- muP recipe and scaling rules
|   |-- safety_policy.md
|   |-- stage3_rust_migration_plan.md <- data-pipeline acceleration plan
|   `-- governance/
|       |-- README.md
|       |-- model_card_v1.md
|       |-- data_card_v1.md
|       `-- license_register.md
|-- scripts/
|   |-- run_pretrain.py              <- main training entry point
|   |-- build_packed_corpus.py       <- per-source corpus builder
|   |-- compose_mixed_corpus.py      <- multi-source mixer
|   |-- run_parallel_builds.py       <- fan-out runner for sources
|   |-- benchmark_throughput.py      <- throughput and MFU bench
|   |-- canary_ladder.py             <- L0/L5 ladder runner
|   |-- canary_l3_resume.py          <- synthetic-data L3 resume test
|   |-- canary_l3_resume_packed.py   <- real packed-corpus L3 resume test
|   |-- train_tokenizer_spm.py       <- SentencePiece tokenizer trainer
|   `-- build_decontamination_index.py <- benchmark decontamination index
`-- src/myllm/
    |-- model/
    |   |-- config.py
    |   |-- transformer.py
    |   `-- layers.py
    |-- training/
    |   |-- train_step.py
    |   |-- loop.py
    |   |-- optimizer.py
    |   |-- mesh.py
    |   |-- checkpoint.py
    |   |-- loss.py
    |   |-- decay_phase.py
    |   |-- watchdog.py
    |   |-- quarantine.py
    |   `-- schedule.py
    |-- data/
    |   |-- packed_corpus.py
    |   |-- compose.py
    |   |-- build.py
    |   |-- loader.py
    |   |-- tokenize.py
    |   |-- synthetic.py
    |   |-- filters.py
    |   |-- decontamination.py
    |   `-- special_tokens.py
    |-- utils/
    |   `-- storage.py
    `-- canary.py
```

---

## 14. POC (proof-of-concept) status — what's been demonstrably proven

| Capability | Status | Evidence |
|---|---|---|
| Streaming HF dataset → tokenized → packed → R2 | ✓ proven | 12 sources × 808.5M tokens uploaded |
| Multi-source mixing with target-share enforcement | ✓ proven | Composed corpus, max drift 1.18% ≤ 2% threshold |
| Tokenizer training at scale | ✓ proven | 131k-vocab SPM-Unigram, ~50GB training corpus |
| Single-GPU training step end-to-end | ✓ proven | L1 canary on 5×H200 |
| Multi-GPU sharded batch (DP-replicated) | ✓ proven | sharding_init data_parallel=5 confirmed |
| Bitwise-exact resume on synthetic data | ✓ proven | L3-synthetic |
| Bitwise-exact resume on real packed corpus | ✓ proven | L3-packed (caught a real prod bug doing it) |
| Chunked tied-LM-head CE | ✓ proven | 6 equivalence tests; matches full CE within 1e-5 |
| Gradient checkpointing per DecoderBlock | ✓ proven | 1B/seq=8192 fits at 61 GB (vs OOM without) |
| Loss-spike watchdog + checkpoint rollback | ✓ unit-tested | Not yet stress-tested in real training |
| Orbax checkpoint save + R2 mirror | ✓ proven | 2.7 GiB at 744 MiB/s |
| R2-streaming corpus build with delete-local | ✓ proven | 808M-token build on 384GB-disk server |
| **FSDP / ZeRO-3 sharded state** | **✓ proven** | **Gauntlet G2: reduce_scatter=46 in HLO (not silent DDP); G4: 1617 MB FSDP vs 14803 MB DP = 89% savings on 2× H200 SXM** |
| **Multi-GPU loss parity (L2 canary)** | **✓ proven** | **DP vs FSDP loss curves match within atol=5e-3 over 50 synthetic steps** |
| **Dual-mode decontam (8+13 gram MinHash)** | **✓ proven** | **10 benchmarks indexed (1.75M / 1.74M ngrams); 558 docs flagged in codeparrot build (real catch)** |
| **Code-only source build** | **✓ proven** | **180M tokens from codeparrot/github-code-clean, 2 shards, R2-hosted (codeparrot fallback for gated starcoderdata)** |
| **Teacher-audit machinery (top-K mass)** | **✓ proven** | **OLMo-2-13B + DeepSeek-V2-Lite loaded on H200 SXM, audit forward ran, top-K masses computed. K decision deferred until re-run on real text (synthetic random tokens → uniform softmax → K=32 recommendation is a methodology artifact, not real)** |
| Reshard checkpoint across mesh shapes (G6) | ◐ partial | Orbax restore call lacks RestoreArgs.sharding; correctness logic exists, API call needs fix before Stage 2 pod transitions |
| Real training at 1B scale | ◯ not yet | Unblocked: Stage 1 pilot can launch |
| Distillation teacher cache build | ◯ not yet | Audit machinery ready; cache step pending Stage 3 prep |
| Eval pipeline run end-to-end | ◯ not yet | Eval adapters exist; not yet wired into training loop |
| Release scorecard | ◯ not yet | Format defined; depends on a v1 model |

**Translation (2026-05-13):** the data + canary + single-step infra is **done and proven**, and the **FSDP correctness gates are green**. We are unblocked for Stage 1 pilot training. The remaining items are operational (compose v2 corpus, run pilot, run rehearsal, run base) rather than design or correctness.

---

## 15. Open decisions (need explicit calls before Stage 3)

1. **seq=4096 vs seq=8192 for base v1.** ◯ STILL OPEN. Depends on FSDP throughput on real training. Decision: re-bench after Stage 1 pilot, pick the one with better cost/wall-time.
2. ~~**Code source: starcoderdata vs codeparrot.**~~ ✓ DECIDED 2026-05-13 — codeparrot/github-code-clean, 180M tokens built and uploaded to R2 at `corpus_v1/sources/codeparrot-github-code-clean/`. Compose-time swap: edit `pretrain_mix.yaml` to replace `bigcode/starcoderdata` with `codeparrot/github-code-clean` in the code slot.
3. **StackExchange formatting.** ◯ STILL OPEN. Current text_field=question wastes the answer. Need to verify the exact preferences-dataset schema (question + chosen_response). Tracked as a Phase 2 rebuild.
4. ~~**Decontam policy: 8gram-only / 13gram-only / dual-mode.**~~ ✓ DECIDED 2026-05-13 — dual-mode (8+13 gram) wired through `build_packed_corpus.py`; both indexes built and on R2. 558-doc catch in the codeparrot build validated the wiring.
5. **Teacher K (top-K cache budget).** ◯ STILL OPEN. Audit machinery is green but the synthetic-random-tokens corpus produced a meaningless K=32 recommendation. Need to re-run on a real-text slice (each teacher's own tokenizer) before locking K. Not blocking Stage 1 (pilot is CE-only); blocking Stage 3 (where teacher cache feeds distillation).
6. **Pod selection for Stage 3.** ◯ STILL OPEN. 2× H200 SXM validated for FSDP correctness; B200 cluster pending. Cost vs throughput tradeoff still TBD; decide after Stage 2 rehearsal's measured throughput.
7. **Eval suite scope for v1.** ◯ STILL OPEN. Full reviewer list (11 benchmarks) vs minimal MMLU/GSM8K/HumanEval. Depends on time/budget. Note: MILU is gated on HF and requires `harshit.hv@samatva.com` to request access at https://huggingface.co/datasets/ai4bharat/MILU before it can be added to the decontam index or eval gate.

---

## 16. Quick glossary

| Term | Meaning |
|---|---|
| **MFU** | Model FLOPs Utilization — achieved FLOPs / peak FLOPs |
| **FSDP** | Fully Sharded Data Parallel — shard params/grads/opt-state across N devices |
| **ZeRO-N** | DeepSpeed's sharding levels (Z1=opt state, Z2=+grads, Z3=+params) |
| **muP** | Maximal Update Parameterization — HP transfer from small to large |
| **WSD** | Warmup-Stable-Decay LR schedule (SmolLM2 / MiniCPM standard) |
| **WSM** | WSD-Stable-Merge — checkpoint averaging at end-of-stable |
| **GQA** | Grouped-Query Attention — fewer KV heads than query heads |
| **YaRN** | Yet another RoPE extension method (long-context, post-training) |
| **L3 / canary** | Verification tier; L3 = forced-kill resume bitwise-exact |
| **R2** | Cloudflare's S3-compatible object store |
| **Chinchilla** | DeepMind's compute-optimal token-per-param ratio (~20:1) |
| **muP base_width** | The reference width at which HPs were tuned |
| **Decay phase** | Last 15% of training where teacher distillation kicks in |

---

## 17. How to read this project tomorrow

If you're picking this up cold and want to be productive in an hour:

1. **Read this file** (~20 min)
2. **Skim `configs/base_1b.yaml`** — that's the spec of what we're training
3. **Skim `docs/governance/model_card_v1.md`** — canonical model decisions
4. **Run** `pytest tests/ -q` — confirms the codebase is healthy (~610 tests passing as of 2026-05-13)
5. **Run** `python scripts/canary_ladder.py --model-config configs/pilot_250m.yaml --tokenizer-path artifacts/tokenizer_v1.json` — confirms L0/L5 pass on the existing corpus

Then for active development, two paths:

**GPU available** (preferred — most remaining work is multi-GPU):

```bash
# Bring up the pod (3 commands, ~15-20 min installs):
git clone --depth 1 https://github.com/cs2hvh/ai-llm.git /workspace/llm-build
cd /workspace/llm-build
bash scripts/pod_setup_apt.sh   # system pkgs: nano, jq, htop, awscli v2
# export R2 + HF + AWS_DEFAULT_REGION=auto then:
bash scripts/pod_launch_gpu.sh   # venv + jax[cuda12] + torch + verify

# Validate FSDP (10-15 min, all 6 gates):
bash scripts/run_fsdp_gauntlet.sh   # G1-G4 should be PASS; G5/G6 are operational

# Then:
#   - compose v2 corpus (codeparrot + others) via scripts/compose_mixed_corpus.py
#   - launch Stage 1 pilot via scripts/run_pretrain.py --model-config configs/pilot_250m.yaml
```

**CPU only:**
- Run the test suite (`pytest tests/ -q`), edit per-source corpus configs, prep the audit-real-text rebuilder, prep the eval gate wiring, prep release-scorecard scaffolding.

The major systems work (FSDP, decontam, code source, audit machinery) is **done**. The remaining work is operational: compose the v2 corpus, run the three training stages, score the release.

---

## 18. Maintenance protocol for this document

Update this file when:
- A phase completes (move it from ◐ to ✓)
- A new bug class is caught (add to risk register)
- Throughput numbers change meaningfully (update §9)
- The plan changes (update §10)
- A new tool or dep is adopted (update §3)
- A file moves (update §13)

The doc is meant to be **the first thing a new contributor reads** and **the last thing the lead checks before sleep**. Keep it tight, current, and honest.
