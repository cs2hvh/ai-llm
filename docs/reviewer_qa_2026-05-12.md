# MyLLM Team Follow-up Q&A — 2026-05-12

External reviewer's reply to the 7 follow-up questions sent after Phase B
re-audit fixes. Source PDF: shared via chat 2026-05-12 PM (not committable
as a binary; this markdown is the canonical record per the
"every load-bearing decision goes into git" governance rule).

Reviewer framing: written as an engineering decision memo — practical
gates, production patterns, concrete thresholds, and where assumptions
are based on public references vs recommendation.

---

## Executive summary (locked decisions)

| # | Topic | Locked decision |
|---|---|---|
| 1 | Pre-tokenization | Sharded CPU worker fleet, target **5M–20M tok/sec aggregate**; Rust tokenizers `encode_batch()`; not single-machine HF map. |
| 2 | Distillation canary | Matched A/B (CE-only baseline vs CE+KL) with **8 gates** including KL trend, gradient sanity, validation CE, gold-token CE, eval delta, style leakage, distribution drift, tail-mass audit. |
| 3 | 1T release | Three decisions at 1T (internal release / external release / continue-to-3T) with a **weighted scorecard** rather than a single benchmark. |
| 4 | Offline corpus | **uint32 tokens** (NOT uint16 — 131k vocab needs 17 bits). **512M tokens/shard**. `tokens.bin` + `seq_meta.arrow` + `doc_meta.parquet` + `manifest.json`. Simple seek index. |
| 5 | Solo-lead process | Canary ladder with **forced-kill resume** is the single most important process control. |
| 6 | H200 throughput | Plan **280K–360K tok/sec aggregate** on 8× H200 SXM; 520K is stretch, not baseline. |
| 7 | WSM | Merge by **duration** (75B/150B/250B token windows), not checkpoint count. Don't cross CE/decay boundary by default. |

---

## 1. Pre-tokenization throughput

Production pretokenization should be sharded, embarrassingly parallel CPU
job: split corpus into medium shards → identical workers over disjoint
shard ranges → deterministic outputs → per-shard task-completion markers.

**Pattern reference**: DataTrove (HuggingFace) — supports local / Slurm /
Ray executors; works with local + S3-style storage via fsspec; records
task-completion markers for failed-job recovery.

```
source shards
 -> worker pool: filter + decontam + tokenize + pack + provenance
 -> output packed token shards + metadata + manifest
```

| Aggregate throughput | Wall time for 1T tokens | Decision |
|---|---|---|
| 5M tok/s | ~55.6 hr | Good enough for 1T if reliable |
| 10M tok/s | ~27.8 hr | Strong target |
| 20M tok/s | ~13.9 hr | Excellent; I/O / provenance likely becomes bottleneck |
| **<1M tok/s** | | Too slow; fix pipeline architecture |

**Tokenizer hierarchy:**
1. Best: Rust `tokenizers` `encode_batch()` inside many CPU workers
2. OK: native SentencePiece C++ inside many CPU workers
3. Prototype only: Python-level per-document loop

HuggingFace `datasets.map(num_proc=...)` is fine for prototyping but not
the production system — less clean for multi-source manifests, per-source
provenance, failure recovery, per-task stats, R2 shard management.

If aggregate throughput is <1M tok/s, the answer is not changing the LLM
architecture — it's moving to batched Rust tokenization, more workers,
better source sharding, less JSON overhead.

---

## 2. Distillation canary pass criteria

Reviewer kept the original 4 gates (finite KL gradient, end-loss <
CE-only baseline, MMLU-ProX Δ ≥ +0.5, nan_skipped < 0.1%) but flagged
them as **incomplete**. The literature shows distillation isn't just
"lower loss" — capacity gap, teacher/student mismatch, bias amplification,
and hidden behavioral transfer (subliminal learning) can all matter.

**Recommended 20B-token canary**: two matched runs with identical data
order / tokenizer / model config / LR schedule / token count:
- **A**: CE-only baseline
- **B**: CE + DeepSeek-V4-Pro-Base top-K distillation

**Full gate set:**

| Gate | Pass threshold |
|---|---|
| Numeric stability | `nan_skipped < 0.1%`; no repeated source-local NaNs |
| KL sanity | KL finite, non-zero, declining over time |
| Gradient sanity | KL grad norm not near-zero AND not >2-3× CE grad norm for long stretches |
| Loss | Validation CE ≤ CE-only baseline by **0.01–0.03 nats** after 20B |
| Gold-token CE | Gold CE does NOT worsen while KL improves |
| Eval | MMLU-ProX / Belebele / HumanEval+ ≥ CE-only baseline; +0.5 nice but noisy at 20B |
| Style leakage | No increase in teacher-name strings, DeepSeek-specific patterns, `<think>` artifacts, refusal/persona markers |
| Distribution drift | Top-token entropy, EOS rate, Hindi token rate, code-token rate remain close to CE baseline |
| Tail audit | Sample top-K mass from full teacher logits; if K=8 top mass <0.95 often, **move to K=16** |

**Most important canary metric isn't MMLU-ProX +0.5** — at 20B tokens,
benchmark deltas are noisy. The signal that matters:

> KL is helping validation CE without causing behavioral/style drift.

**Failure modes where the original 4 gates pass but the model is still
poisoned:**
1. Teacher distribution over-regularizes the student (less diverse generation)
2. Top-K truncation hides tail behavior on code/math/Hindi
3. Domain-specific leakage (teacher-styled on code/math/safety)
4. Latent behavioral transfer (subliminal learning — less likely cross-base than same-base, but worth probing)

---

## 3. 1T release decision threshold

**Reframing:**
> 1T is not judged by "did the loss flatten?"
> 1T is judged by "does this checkpoint meet product/release gates, and is more training still high-ROI?"

Loss almost never fully flattens — you stop because marginal $/nat drops,
or because the model clears the release bar. Public small-model precedent
supports the "1T internal v1" posture: Llama 3.2 1B/3B = 9T, OLMo 2 1B
= 4T, SmolLM3 3B ≈ 11.2T.

### Three decisions at 1T

| Decision | Ship / continue rule |
|---|---|
| Internal validation release | Ship internally if: training was stable, resume works, data/provenance/governance complete, no major contamination/legal issue, evals directionally sane |
| External base-model release | Ship externally only if: **MMLU ≥ 42, Belebele En ≥ 50, Belebele Hi ≥ 35, HumanEval+ ≥ 15** |
| Continue to 3T | Continue if validation loss + evals still improve materially, OR if Hindi/code/math below product floor while their validation loss is still dropping |

### Continue-to-3T signals

| Signal | Decision |
|---|---|
| Validation loss improves >0.01 nats per 100B over the last 200B | Continue to 3T |
| MMLU / MMLU-Pro / MMLU-ProX still improve checkpoint-to-checkpoint | Continue to 3T |
| Code/math evals lag while code/math validation loss still drops | Continue OR do targeted code/math continued pretrain |
| Hindi/Indic below product floor | Continue, but consider Indic-heavy continued pretrain over generic 3T |
| Model clears product floor AND loss gain <0.005 nats per 100B | Maybe stop |
| Governance/data/legal incomplete | **Do not externally release regardless of eval** |

### Weighted release scorecard

Don't use one benchmark as the release switch:

| Category | Weight |
|---|---|
| General reasoning | 40% |
| Code/math | 20% |
| Hindi/Indic + multilingual | 20% |
| Safety/memorization | 10% |
| Serving/cost/governance | 10% |

---

## 4. Offline corpus shard design

**Load-bearing correction:** token IDs CANNOT fit in 2 bytes. 131,072
vocab → max token id 131,071 → needs 17 bits. uint16 only addresses 0–65,535.
**1T × 2 bytes/token is invalid math** unless vocab is changed or custom
bit-packing is implemented (don't).

### Storage budget (uncompressed)

| Component | Rough size |
|---|---|
| Token IDs, **uint32** | ~4.0 TB |
| Segment/loss metadata if stored separately | Avoid if derivable |
| Per-sequence metadata | ~50–300 GB |
| Per-doc provenance table | ~50–500 GB |
| Manifest / index / checksums | < 10 GB |
| **Total realistic uncompressed** | **~4.5–5.5 TB** |
| With compressed metadata, raw tokens uncompressed | ~4.2–5.0 TB |

### Storage layout (per shard)

```
tokens.bin       uint32 memmap
seq_meta.arrow   compact per-packed-sequence metadata
doc_meta.parquet compressed source/doc provenance (dictionary-encoded)
manifest.json    top-level
```

### Seek index requirement

Yes — a seek index IS needed; do not rely on `skip_first N docs`.
Streaming datasets aren't designed for token-exact random access.

Because packed sequences are fixed length, the index can be simple math:

```
sequence_id -> shard_id, sequence_offset

shard_id     = sequence_id // sequences_per_shard
offset       = sequence_id %  sequences_per_shard
byte_offset  = offset * packed_sequence_bytes
```

### Two-level shard design

| Level | Purpose | Example |
|---|---|---|
| Build-time source shards | Preserve source-local filtering, revision pins, quality stats, provenance | `source/fineweb_edu/...`, `source/stack_v2/...`, `source/sangraha_hi/...` |
| Training-time mixed packed shards | Optimize resume/training throughput; enforce target token distribution | `train/shard_000000.tokens.bin`, `train/shard_000000.seq_meta.arrow` |

Do not train from pure source shards unless an explicit curriculum is
intended — pure source shards make per-source validation easier but make
the pretraining distribution less smooth.

### Shard size

| Shard size | Approx raw size | # shards for 1T |
|---|---|---|
| 100M tokens | ~400 MB | ~10,000 |
| 256M tokens | ~1 GB | ~3,907 |
| **512M tokens** | **~2 GB** | **~1,954** |
| 1B tokens | ~4 GB | ~1,000 |

**Recommended: 512M tokens/shard for 1T.**

### Metadata schema

**Per-packed-sequence (seq_meta.arrow):**
```
sequence_id
token_start_global
token_end_global
source_mix_histogram
doc_span_start_id
doc_span_count
```

**Per-doc-span (doc_meta.parquet, dictionary-encoded):**
```
doc_span_id
sequence_id
source_id
doc_id_hash
dataset_revision_id
token_start_in_sequence
token_end_in_sequence
text_hash
```

Avoid huge raw JSONL for 100M+ packed sequences — Arrow/Parquet with
dictionary encoding is the production format.

---

## 5. Solo-lead failure pattern

The failure pattern for solo-led LLM builds isn't "not enough unit tests"
— it's:

> unit tests validated components, but no test validated the full training contract

The process control that saves the project is a **canary ladder with
hard acceptance gates**.

### Most important process rule

Before any real run, require the same command path, same config resolver,
same data path, same checkpoint path, same resume path, same logging path,
and same evaluation path on a tiny version of the run. **A unit test that
calls a function is not enough — a canary must run the actual training
script.**

### Recommended canary ladder

| Stage | Purpose | Required before |
|---|---|---|
| 100-step synthetic | Forward/backward/optimizer/checkpoint smoke | Any real data |
| 1k-step real data single source | Tokenizer/filter/packer/doc-mask | Wind-tunnel |
| 5k-step mixed data | Mixture, decon, quarantine, resume | Proxy A sweep |
| Forced-kill resume | Checkpoint + data cursor | Proxy B |
| 1B-shape 100-step synthetic | Memory, shapes, compilation, attention | 250M pilot |
| 250M 1B-token real canary | Throughput + data + eval | 30-50B pilot |
| 250M forced-kill mid-run | Operational confidence | 1B base |
| 1B 1B-token canary | Final launch dry run | 1T base |

### Full-scale-only bug classes

10 classes flagged. **All 10 now have regression-test coverage** — see
[`docs/full_scale_bug_coverage_2026-05-12.md`](full_scale_bug_coverage_2026-05-12.md).

> Every bug found during reviews should become a permanent regression test or canary assertion.

---

## 6. Throughput target for 1B on 8× H200 SXM

The 65K tok/s/GPU × 8 = 520K assumption is a **stretch ceiling**, not a
safe cost-model baseline. H200 has the same nominal BF16 tensor peak as
H100 — the H200 advantage is memory capacity + bandwidth, not raw FLOPS.

### Planning range

| Scenario | tok/sec/GPU | 8× aggregate |
|---|---|---|
| Conservative first working run | 25K–35K | 200K–280K |
| **Good optimized BF16 run** | **35K–45K** | **280K–360K** ← planning baseline |
| Strong optimized run | 45K–55K | 360K–440K |
| 65K/GPU assumption | Stretch | ~520K |

### Cost-model implication (1T tokens)

| Aggregate | Time for 1T |
|---|---|
| 200K | ~57.9 days |
| **280K** | **~41.3 days** |
| 360K | ~32.2 days |
| 440K | ~26.3 days |
| 520K | ~22.3 days (don't plan against) |

**Use 280K–360K aggregate as the planning baseline until measured.**
>400K = upside.

### Measurement protocol

- Run a **2-hour throughput benchmark**
- Use the **real packed corpus**, not synthetic data
- Enable checkpointing, logging, doc-mask, same context length, same
  micro-batch, same grad accumulation, same optimizer, same sharding
- **Synthetic data hides dataloader + packing overhead** — don't use it
  for the cost model

---

## 7. WSM merge count and duration

Key WSM idea: merge by **duration**, not by checkpoint count.

For a 1T MyLLM run with ~150B-token decay phase, test multiple WSM
windows from the stable phase:

| Candidate | Window | Checkpoints | Spacing |
|---|---|---|---|
| WSM-small | Last 75B stable tokens | 4-5 | ~15-20B |
| WSM-mid | Last 150B stable tokens | 6 | ~25B |
| WSM-large | Last 250B stable tokens | 6-8 | ~35-40B |

Evaluate all three against:
- stable-final single checkpoint
- WSD-decayed final
- distilled-decay final

### Interaction with distillation

**Do not blindly average checkpoints across the CE-only / decay-phase
boundary.** They optimize different objectives:
- Stable: CE-only
- Decay: CE + KL

Averaging across is less principled — treat as an experiment.

**For v1 protocol:**
1. Run WSM over CE-only stable checkpoints
2. Evaluate the WSM checkpoint
3. Optionally run a short decay/distillation branch from the best single
   stable checkpoint AND from the best WSM checkpoint
4. Compare to answer: does WSM help before distillation? Does distillation
   still help after WSM?

---

## References cited by reviewer

- HuggingFace Tokenizers documentation — Rust-backed production tokenizer performance
- HuggingFace Transformers fast tokenizer documentation — batched tokenization
- HuggingFace Datasets — `.map(num_proc=...)`, streaming
- DataTrove — scalable data processing, Ray/Slurm/local executors
- MosaicML StreamingDataset — checkpoint/resume concepts
- Springer 2026 KD Survey — risks + KD limitations
- Anthropic alignment blog — Subliminal learning
- Nature article — Subliminal learning in language models
- Meta Llama 3.2 model card — training-token anchor
- NVIDIA H200 product page — specs
- OLMo-core PyPI — throughput reference
- NVIDIA Megatron-LM — MFU and optimized training
- WSM (OpenReview / ICLR 2026, id=HhThhjKyfw) — merge duration over checkpoint count

---

## Implementation status (as of 2026-05-12 PM)

| Decision | Status in repo |
|---|---|
| uint32 tokens, 512M-token shards | ⏳ B2 work — design locked, code pending |
| Sharded CPU worker pretokenizer | ⏳ B2 work |
| Distillation canary harness (matched A/B + 8 gates) | ⏳ Phase C |
| 1T weighted scorecard + thresholds | ⏳ Phase 3 (codified in `docs/governance/eval_card_v1.md` when v1 ships) |
| 280-360K H200 planning baseline | ✅ saved to `memory/feedback_h200_throughput_baseline.md` |
| WSM by duration (75/150/250B) | ⏳ Phase 3 |
| Canary ladder + forced-kill resume | ⏳ scripts/canary_ladder.py (pending) |
| 10 full-scale bug classes → regression tests | ✅ done, see [`full_scale_bug_coverage_2026-05-12.md`](full_scale_bug_coverage_2026-05-12.md) |
