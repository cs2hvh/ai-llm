# MyLLM Plan v3 — After Review 3 (2026-05-12)

**Status:** decision record. Supersedes Phase 2-3 sections of `project_handoff_2026-05-11.md`.

This plan integrates: the third external review (review3.pdf, 2026-05-12), the
reviewer follow-up Q&A (`reviewer_qa_2026-05-12.md`), the full-scale-only bug
coverage work (`full_scale_bug_coverage_2026-05-12.md`), and a 12-agent
parallel research pass on scaling laws, frontier-lab recipes, hardware
throughput, distillation A/B protocols, and governance requirements.

---

## 1. Trust assessment of review 3

**Verdict: trust + act.** Every load-bearing claim independently verified.

**Local code claims (7/7 confirmed):**

| # | Claim | Verified |
|---|---|---|
| 1 | `context_extension_yarn_target` schema mismatch with `context_extension_target` | ✅ value silently stored under wrong key |
| 2 | `mesh.py` says FSDP "planned for the next iteration" | ✅ verbatim line 9 |
| 3 | `run_pretrain.py` docstring/imports reference cosine-warmup; live code is WSD | ✅ line 7 + dead `cosine_with_warmup` import line 77 |
| 4 | `loader.py` docstring claims "per-shard byte-offset checkpointing" but code is `skip_first` counter | ✅ lines 5 vs 48, 86 |
| 5 | `MixtureSampler` default measure is char-length, not tokens | ✅ `_default_measure` returns `len(text)`; doc admits "~4 chars/token proxy" |
| 6 | `dedupe.py` exists but is NOT wired into the build path | ✅ zero imports in `run_pretrain.py` or `loader.py` |
| 7 | base_1b.yaml `expected_cost_usd_baseline_1T: 240000` is stale | ✅ off by ~7-10× from current planning numbers |

**External claims (verified via parallel research):**

| Claim | Verdict | Source |
|---|---|---|
| Llama 3.2 1B "up to 9T tokens" | ✅ VERIFIED | HF model card verbatim |
| Llama 3.2 1B used 8B + 70B logits **during pretraining** | ✅ VERIFIED — pretraining-phase, not post-training | HF card + Meta blog |
| Llama 3.2 chat = SFT + DPO-like | ⚠️ PARTIAL — actually **SFT + Rejection Sampling + DPO**, multi-round | HF card |
| OLMo 2 1B "4T tokens" | ✅ VERIFIED — 4T stage-1 + 50B Dolmino mid-training | arXiv 2501.00656 |
| SmolLM3 "11T-11.2T at 3B" | ✅ VERIFIED — 11.2T (8T+2T+1.1T WSD) + 140B mid-train | HF blog |
| 280-360K tok/s on 8× H200 SXM | ⚠️ OPTIMISTIC at seq=8192 — realistic 240-280K | MosaicML measured 233K @ 8×H100 |
| EU AI Act systemic-risk = 10²⁵ FLOPs | ✅ VERIFIED | EU AI Act Art 51(2) |

---

## 2. New material findings beyond review 3

These are facts the research surfaced that change planning assumptions:

### 2.1 Llama 3.2 1B is NOT a fair from-scratch comparable

Meta's HF card verbatim: *"For the 1B and 3B models, we took the approach of using structured pruning in a single shot manner from the Llama 3.1 8B."* Llama 3.2 1B is:
- Pruned from 8B (warm-start weights, not random init)
- Distilled from 8B + 70B logits **during pretraining itself** (every-token supervision throughout, not just decay phase)
- Then SFT + Rejection Sampling + DPO across multiple rounds for Llama 3.2 1B-**Instruct**

**Implication:** "1B at 1T tokens, from-scratch, decay-only distillation" cannot reach Llama 3.2 1B parity by design. The only true from-scratch 1B comparable in this size class is **OLMo 2 1B at 4T tokens**.

### 2.2 Distillation Scaling Laws (Busbridge et al., arXiv:2502.08606)

Verbatim: *"Supervised learning always outperforms distillation given enough student compute or tokens."* Distillation is a **finite-budget win**, not free uplift.

Concrete data point: **Gemma 2 2B distilled from 7B teacher = +7.4% over CE-only at matched token count** (arXiv:2408.00118 §6). Roughly **1.5-2× effective token multiplier**, not 9×.

### 2.3 EU AI Act compliance posture (mid-2026)

- Threshold for "GPAI with systemic risk" = **10²⁵ training FLOPs** (Art. 51(2))
- MyLLM 1B × 1T tokens = **6×10²¹ FLOPs ≈ 1,700× below threshold**
- **Standard GPAI obligations apply; no systemic-risk obligations**
- **Open-weight Art. 53(2) exemption** drops Annex XI/XII technical documentation, but **training-data summary (Art. 53(1)(c)) and copyright policy (Art. 53(1)(d)) are non-waivable**

### 2.4 FSDP is NOT needed at 1B scale

MosaicML measured: 1.316B model, bf16 mixed precision, seq=8192, full replication, **µBS=3 fits in H100 80GB**. H200 141GB has 60GB headroom → **µBS=6-8 with pure DP replication**. FSDP becomes necessary at ~7B+. `mesh.py`'s "FSDP planned for next iteration" comment is **aspirational but operationally moot** at our scale.

### 2.5 Dedupe must be per-source, NOT global

FineWeb's explicit finding (arXiv 2406.17557): global dedup across 96 CC dumps retained only ~10% of data and **that retained slice was lower quality**. Production recipe:
- Exact URL dedup: global (cheap)
- Document-level fuzzy dedup: per-snapshot / per-source
- MinHash+LSH: **112 hashes × 14 bands × 8 rows, ~0.75 Jaccard threshold** (DataTrove/FineWeb default)

### 2.6 Pretokenization is not a bottleneck

One hpc7a.96xlarge (192 EPYC cores, ~$7/hr) tokenizes 1T tokens in ~10-15 hours for **<$200**. R2 PUT throughput will be the actual bottleneck. Avoid `datasets.map(num_proc=...)` — documented tail collapse from 19-37K examples/s → 36-185 examples/s near completion (HF issue #6734).

### 2.7 Realistic 1B-on-8×H200 throughput at seq=8192 = 240-280K tok/s

The "280-360K planning baseline" in earlier memos is realistic to optimistic. **240-280K is the conservative number**; 360K requires shorter context (4k) or near-best-case kernel tuning.

### 2.8 Provider economics (mid-2026)

| Topology | Provider | Wall time | Cost (1T tokens) |
|---|---|---|---|
| 8× H200 SXM (single pod) | Lambda 1-Click | ~43 days | **~$31-37K** |
| 32× H200 SXM (4 pods, IB) | Lambda 1-Click | ~7 days | **~$22K** |
| 32× H200 spot | CoreWeave | ~7 days | ~$15K (high preemption risk) |
| 32× B200 SXM | Lambda 1-Click | ~4 days | ~$19K |

**Multi-pod 32×H200 is BOTH faster AND cheaper** than single-pod 8×H200 (better scaling efficiency despite more GPUs because per-GPU cost is constant and total GPU-hours drop with linear scaling). Reliability risk on multi-pod is real (IB link flaps, NCCL hangs) but mitigable with Lambda's validated fabric.

### 2.9 Canary ladder catches real published bugs

Forced-kill resume must round-trip exactly: optimizer state (m, v, step), fp32 master, data cursor (shard+sample+intra-sample offset), **teacher cache cursor hash**, per-rank RNG, LR scheduler step. Documented failure modes:
- Tied-embedding gradients not reduced (BLOOM tr11)
- LayerNorm in weight-decay group → checkpoint mismatch on reload (BLOOM)
- FFN_HIDDEN_SIZE typo silently shipping wrong param count (BigScience tr8)
- Loss spike from data-batch / optimizer-state interaction (OPT-175B, 35 restarts)

### 2.10 Per-source validation + memorization probes

- Per-source val: **8M tokens/source flat (NOT scaled by mix weight)**, every 1000 steps. 13 sources × 8M = ~100M val tokens total.
- Memorization: Pythia (k=50, l=50)-extraction protocol on ~100k random training sequences/source. **>2% extraction rate = release-blocker.**
- Canary insertion: 100 random 9-digit nonces at log-spaced frequencies (1×, 4×, 16×, 64×, 256×). **Any canary inserted ≤4× with exposure >20 = release-blocker.**

---

## 3. Capability framing (revised)

**Drop:** "Llama 3.2 1B parity at 1T."

**Adopt:** "MyLLM-1B internal-v1 = OLMo 2 1B-class quality on roughly 1/4 the token budget, with decay-phase distillation closing ~1.5× of the gap."

Realistic 1T outcome (estimated, no controlled study at this exact point):
- MMLU 38-42 (vs OLMo 2 1B ~42, Llama 3.2 1B ~49)
- Belebele En 45-52 / Hi 30-37
- HumanEval+ 12-18
- GSM8K 25-35

**Continue-to-3T is the expected path**, not the fallback. OLMo 2 / SmolLM3 / Qwen3 all extended pretraining past initial budget. Plan for it from day one.

---

## 4. Prioritized roadmap

### Week 1 — P0 drift fixes (1-2 days, no GPU spend)

1. Fix `context_extension_yarn_target` → `context_extension_target` in `configs/base_1b.yaml`
2. Update `expected_cost_usd_baseline_1T: 240000` → realistic range ($22K multi-pod / $33K single-pod)
3. Update `loader.py` docstring to match code (it's `skip_first`, not byte-offset checkpointing)
4. Update `run_pretrain.py` docstring (WSD not cosine) + remove dead `cosine_with_warmup` import
5. Add note to `MixtureSampler` module docstring on chars-vs-tokens limitation (full fix lands with B2)

### Weeks 2-4 — B2 offline corpus

- uint32 token shards, 512M tokens/shard, ~1,954 shards for 1T
- Layout per shard: `tokens.bin` + `seq_meta.arrow` + `doc_meta.parquet` + `manifest.json`
- Native SentencePiece tokenizer via multiprocessing on one hpc7a.96xlarge (~$200, ~15h)
- **Per-source MinHash+LSH dedupe (NOT global)**, 112×14×8, ~0.75 Jaccard threshold — wire `dedupe.py` into the build path
- Revision pinning per source (HF dataset commit SHA in manifest)
- Per-doc provenance: `(source_id, doc_id_hash, dataset_revision_id, token_start_in_seq, token_end_in_seq, text_hash)`
- Manifest carries: tokenizer SHA256, source revisions, **exact token-share per source** (now trivial since shards are pre-tokenized), build timestamp
- Estimated cost: **<$300** (compute + R2 storage for ~4 TB)

### Week 5 — Canary ladder (no GPU pod time yet)

- **L0**: param-count exact, tokenizer round-trip lossless
- **L1**: 20-step single-GPU loss-decreasing smoke (matches OLMo-core pattern)
- **L2**: 8-GPU multi-device parity test (loss matches single-GPU within 1e-6 fp32 master)
- **L3**: forced-kill resume bitwise-exact (loss at step N matches uninterrupted run to ≤1e-4 bf16 noise floor; hash of (data cursor, params, optimizer m/v) at step N matches)
- **L4**: 1B-shape 1-2% scale rehearsal on real packed data, sustain ≥35% MFU for 1 hour
- **L5**: data sanity (per-source histograms, teacher cache cursor alignment if distillation cache exists)

### Week 6 — Re-launch Proxy A + Proxy B transfer validation

- Proxy A 10-cell sweep at 67M, 200M tokens/cell (~$30-50)
- Proxy B single cell at 300M, width_mult=4, 500M tokens at (LR*, init*) (~$11-20)
- Pass: smooth loss curve, ≤0.2 nats from muP-predicted end-loss

### Weeks 7-9 — 250M pilot + A/B distillation canary

- Pilot 250M: 30-50B tokens on packed corpus, ~$500-1K
- **Matched A/B canary** (5-10B tokens each arm):
  - Arm A: CE-only baseline
  - Arm B: CE + KL with DeepSeek-V4-Pro-Base (1 teacher only for canary)
- DeepSeek-V4 cache for canary: ~$500
- **9 pass gates** (8 from reviewer + 1 from research):
  - `nan_skipped < 0.1%`
  - KL declining, finite, non-zero
  - Gradient norm parity (KL grad ≤ 2-3× CE grad)
  - Val CE: CE+KL ≤ CE-only - 0.01-0.03 nats
  - Gold-token CE not worsening while KL improves
  - Eval (MMLU-ProX / Belebele / HumanEval+) ≥ baseline
  - Style leakage: no rise in teacher-name / `<think>` markers
  - Distribution drift (top-token entropy, EOS rate, per-domain rates) within 5% of CE-only
  - **NEW from research**: Reverse-KL on student-sampled prefixes trending down (Thinking Machines on-policy diagnostic — flat = compounding error)
  - **Tail-mass audit**: if K=8 misses >5% of teacher mass on code/math sampling, **bump to K=16 BEFORE full cache generation**

### Weeks 10-16 — 1T base run

**SKU decision**: 32×H200 SXM via Lambda 1-Click Clusters (4 pods of 8, Quantum-2 IB).
- Wall time: ~7 days
- Cost: **~$22K**
- 90% stable + 10% decay (~100B decay tokens)
- Decay-phase data mix: more code, math, Hindi (per OLMo 2 Stage-2 pattern)
- Per-source val every 1000 steps, 8M tokens/source
- Memorization probes at 50%, 75%, 100% milestones
- Checkpoint every 500 steps, R2 mirror every successful save

Alternative if compute availability blocks multi-pod: 8×H200 single pod, ~43 days, ~$33K. **Costlier and slower** but operationally simpler.

### Week 17+ — Release decision via weighted scorecard

**Internal v1 release threshold:**
- Training was stable, resume tested, no major NaN events
- Per-source val curves all monotone-decreasing in stable phase
- No release-blocking memorization probe failures
- Governance artifacts complete (model card, data card, license register, eval report)

**External release gate** (weighted scorecard, reviewer's framework):
| Category | Weight | Floor |
|---|---|---|
| General reasoning | 40% | MMLU ≥ 42 |
| Code/math | 20% | HumanEval+ ≥ 15, GSM8K ≥ 30 |
| Hindi/Indic + multilingual | 20% | Belebele En ≥ 50, Hi ≥ 35 |
| Safety/memorization | 10% | <2% extraction rate, no canary exposure >20 |
| Serving/cost/governance | 10% | TEVV report complete |

**Continue-to-3T trigger:**
- Validation loss > 0.01 nats/100B improvement over last 200B
- Eval scores still climbing checkpoint-to-checkpoint
- Hindi/code below floor while their domain val loss is still dropping

---

## 5. Decision-grade cost envelope

| Phase | Item | Cost |
|---|---|---|
| W1 | P0 drift fixes | $0 |
| W2-4 | B2 corpus build (compute + R2 storage) | ~$300 |
| W5 | Canary ladder | $0 (CPU + 1-GPU smoke) |
| W6 | Proxy A re-run + Proxy B | ~$50-70 |
| W7-9 | 250M pilot + A/B canary + DeepSeek-V4 cache for 20B canary | ~$1.5-3K |
| W10-16 | **1T base run on 32×H200 (Lambda)** | **~$22K** |
| W10-16 alt | 1T base run on 8×H200 single pod | ~$33K |
| Ongoing | R2 storage (4 TB corpus + 7 TB teacher cache + ~50 checkpoints) | ~$200-400/year |
| **Phase 2-3 total (multi-pod path)** | | **~$25K** |
| **Phase 2-3 total (single-pod path)** | | **~$36K** |

The stale `$240K` in base_1b.yaml was off by ~7-10×.

Continue-to-3T (if triggered): +2× the base run cost = +$22K. Total ~$47K for 1B at 3T.

---

## 6. References

- `docs/reviewer_qa_2026-05-12.md` — prior reviewer Q&A locking B2 design
- `docs/full_scale_bug_coverage_2026-05-12.md` — coverage map for the 10 full-scale-only bug classes
- `docs/external_review_2026-05-12_enterprise.md` — enterprise strategy review
- `docs/MyLLM_Repo_Technical_Review_2026-05-12.docx` — first colleague's code review
- review3.pdf (2026-05-12) — this document's primary source
- 12 parallel research-agent reports (in-chat 2026-05-12, not committed to repo)
