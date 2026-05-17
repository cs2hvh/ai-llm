# MyLLM — Project Review Packet (2026-05-12)

**Reader**: Senior AI researcher (ex-Mistral et al.)
**Author**: harshit.hv (solo lead) + Claude as build partner
**Status**: Phase 2 pilot pre-launch; 1B-token corpus in flight; 8×B200 incoming in ~4-5 hr
**Ask**: full-scope review — please push back on anything that looks wrong. No need to be polite about it.

> The model card, data card, and license register live in [`docs/governance/`](../governance/) — this packet is the **state + validations + open risks** view, not a duplicate of those. Treat the governance docs as the canonical spec and this as the operator's confession of what's been built vs. what's still on faith.

---

## 1. Executive context (one paragraph)

We're training a **1B-parameter from-scratch decoder-only foundation model** ("MyLLM") with a distillation-augmented decay phase. **Pilot = 250M / 50B tokens** on 1×B200 or 8×H100 (config in [`configs/pilot_250m.yaml`](../../configs/pilot_250m.yaml)); **Base = 1B / 1T tokens** on 8×B200 or 8×H200. Tokenizer is **SentencePiece-Unigram 131,072 vocab with byte fallback** (trained natively via SentencePiece, not HF tokenizers — HF tokenizers train-time hit a ~15 GB corpus ceiling). The training stack is **Keras 3 + JAX backend** with sharded Orbax checkpoints, atomic NaN-revert in the train step, a loss-spike watchdog with rollback + LR-recovery multiplier, and a 5-rung canary ladder (L0 static / L1 single-GPU smoke / L3 forced-kill resume bitwise-exact / L5 source-share drift). **The corpus, infra, and canaries are all built and tested in isolation; what's still on faith is the multi-GPU production run and the distillation cache path.**

---

## 2. Current state (as of 2026-05-12, IST evening)

### What's running right now
- **1B-token mixed corpus build** on a 96-vcpu / 384 GB CPU server (Hetzner). 11/12 sources DONE (~369M tokens uploaded to R2). FineWeb-Edu in-flight at ~31K tok/sec, 154M of 440M target packed, ETA ~2.6 hr. Run config: `--target-tokens-per-source 1000000000 --max-parallel 12 --no-decontam --skip-sources bigcode/starcoderdata --delete-local-after-upload`.

### What's validated on H200 SXM today
- **L0 canary** (static): config invariants, tokenizer roundtrip, shape checks. PASS.
- **L1 canary** (single-GPU 20-step smoke, synthetic data, micro_batch=2, model=pilot_250m): forward + backward + AdamW + Orbax checkpoint save (2.7 GiB @ 744 MiB/s). Loss `11.95 → 11.95` (expected — random data, warmup ramp). PASS.
- **L3 canary** (forced-kill resume bitwise-exact, tiny model, CPU): `ref_state_hash == resumed_state_hash`. PASS *after fix*. **See §6 for the bug L3 caught.**
- **Throughput benchmark**, 1×H200 SXM, pilot_250m at micro_batch=4, 20 warmup + 100 measure steps, synthetic data:
  - **47,308 tok/sec/device**
  - **MFU 3.4%** (1979 TFLOPS BF16 peak; achieved ~71 TFLOPS)
  - **Peak HBM 101.5 GB / 141 GB** available
  - Extrapolation: 1T tokens on 8×H200 ≈ **734 hr ≈ 30.6 days ≈ ~$20.5K** at RunPod $3.5/hr/GPU.

### What's NOT yet validated
- Multi-GPU training (FSDP / TP). Single-GPU only so far.
- Real-data (packed-corpus path) training run; only synthetic-data L1.
- Loss-spike watchdog end-to-end (unit-tested, not stress-tested in real training).
- Distillation teacher-cache build + lookup at scale.
- WSM checkpoint averaging in a real training run.
- Decontamination index pre-build path (`scripts/build_decontamination_index.py` exists, hasn't been run on the 1B corpus).
- 1T-token corpus build (this 1B run is a pragmatic interim — 1T is ~40 days on this CPU server because FineWeb-Edu @ 44% × 1T = 30 days at 17K tok/sec ingest).

---

## 3. Architecture decisions (pilot_250m → 1B base)

Full config in [`configs/pilot_250m.yaml`](../../configs/pilot_250m.yaml) and [`configs/base_1b.yaml`](../../configs/base_1b.yaml). Llama-style decoder. Pilot summary:

| Field | Value | Reason |
|---|---|---|
| layers / hidden / ffn | 16 / 768 / 3072 | 4× FFN; matches Llama-3.2-1B's ratio at smaller scale |
| heads / kv_heads | 12 / 4 | GQA 3:1 (Llama-3 ratio) |
| context_length | 4096, extend to 8192 end-of-pretrain | RoPE base 130000 (long-context-friendly per Yarn) |
| vocab_size | 131,072 | SentencePiece-Unigram + byte fallback; multilingual support |
| tie_embeddings | True | 100M embed params on a 250M model — untied would 1.4× the model |
| qk_norm | True | **muP HP transfer requirement**: LR sensitivity differs ~10-30% with QK-norm on/off, exceeds muP's ±10% transfer tolerance |
| z_loss_coef | 1.0e-4 | Standard stability term |
| init_std | 0.02 | scaled_init_for_residuals=False at pilot (enable for base per re-audit) |
| optimizer | AdamW β1=0.9 β2=0.95 wd=0.1 ε=1e-8 | |
| lr_schedule | WSD (Warmup-Stable-Decay), peak 3e-4, warmup 2000, decay last 15% | Per SmolLM2 / MiniCPM playbook |
| mixed_precision | bfloat16 | |
| grad_clip | global_norm = 1.0 | |
| target_tokens | 50B baseline / 100B stretch | Chinchilla-optimal for 250M is ~5B; we're aggressively over-training for distillation efficacy |

**muP transfer plan**: Proxy A (250M tied-embed, Phase B1) → Proxy B (300M, different head count, Phase B2) → 1B base. HP sweep happens at Proxy A; if Proxy B reproduces the sweet-spot within ±10%, we transfer to base.

**Distillation strategy (locked 2026-05-12)** — see [`/root/.claude/projects/-root/memory/project_teacher_strategy.md`](../../../.claude/projects/-root/memory/project_teacher_strategy.md) for full reasoning. Decay-phase teacher distillation (last 15% of training) with:
- Teachers: **DeepSeek-V4-Pro-Base** + **Olmo-3-32B-Base** (averaged top-K logits)
- Dropped during license/modality review: Mistral (license restrictive), Qwen3.6 (modality mismatch)
- Top-K = 64 logits + indices cached at corpus-position granularity
- Stable phase: α=1.0 (pure CE); decay phase: α anneals 0.7 → 0.3

---

## 4. Data pipeline

Full spec in [`configs/data/pretrain_mix.yaml`](../../configs/data/pretrain_mix.yaml). 1B-token corpus shares (44% web / 18% code-slot-pending / 6% wiki / 5% books / 6% academic / 7% math / 2% Q&A / 12% multilingual):

| Source | Share | Status |
|---|---|---|
| HuggingFaceFW/fineweb-edu | 0.44 | ✓ (covers Nemotron-CC slot temporarily) |
| bigcode/starcoderdata | 0.18 | ⚠ **GATED**, skipped in current build; fallback codeparrot/github-code-clean documented |
| wikimedia/wikipedia (20231101.en) | 0.06 | ✓ |
| pg19 | 0.05 | ✓ (re-added after pg19 over-fetch fix; see §6) |
| allenai/peS2o | 0.06 | ✓ |
| open-web-math/open-web-math | 0.07 | ✓ (absorbed proof-pile-2's 4.2% — proof-pile-2 had zstd decompress errors mid-stream) |
| HuggingFaceH4/stack-exchange-preferences | 0.02 | ✓ |
| ai4bharat/sangraha (Hindi only) | 0.04 | ✓ (sovereign-language hedge) |
| mc4 × 5 configs (es/zh/ar/fr/de) | 0.08 total | ✓ (mc4 deprecated, auto-redirects to allenai/c4; works with trust_remote_code) |

**Sources we removed**:
- `nvidia/Nemotron-CC` — pending NVIDIA approval; FineWeb-Edu temporarily covers
- `EleutherAI/proof-pile-2` — zstd decompress errors + multiple config issues
- `bigcode/the-stack-v2` — content stored out-of-band in Software Heritage / S3, not in HF parquet; pipeline can't fetch per-blob without custom loader

**Filter chain** (applied per-document at load time):
- Length: 200 ≤ chars ≤ 1M
- Repetition: top word ≤ 20%, top 5-gram ≤ 10%
- Symbol ratio: ≤ 30%
- PII: email + phone redacted (IPv4 retained — could be code samples)

**Decontamination** ([`src/myllm/data/decontam.py`](../../src/myllm/data/decontam.py)):
- 13-gram MinHash+LSH, hash_seed=0xDECAF
- 11 benchmarks indexed: mmlu-prox, belebele, milu, mmlu-pro, humaneval-plus, mbpp-plus, gsm8k, math, mgsm (per-lang ×7), bbh (per-subtask ×27), ifeval
- **Currently disabled** in this 1B run (`--no-decontam`) for speed; will be enabled for base run with pre-built index
- **Open question for you**: 13-gram is what we settled on. Llama 3 uses 8-gram with higher recall. What's your read on the recall/precision tradeoff at our scale?

**Storage**: Cloudflare R2 with 32-way parallel multipart upload (TransferConfig multipart_chunksize=16MB, max_concurrency=32). Measured 4.7× speedup over single-stream (25.5 MB/s → 119.3 MB/s; sustained 954 Mbps to R2 at corpus build time). `--delete-local-after-upload` keeps the staging filesystem flat — each shard is uploaded synchronously then unlinked locally; in-memory shard manifest is aggregated and written once at corpus close.

**Tokenization**: HF tokenizers Rust core + Rayon parallelism via `encode_batch` (14× speedup over per-doc encode). Char-aware adaptive batch flush (early-flush when `chars/3` lower bound on token estimate would exceed `--target-tokens`); see §6 for why this matters.

---

## 5. Tokenizer

`artifacts/tokenizer_v1.json` — SentencePiece-Unigram, 131,072 vocab, byte fallback. SHA256 cross-checked at every training start (loop refuses to train if the saved corpus's `tokenizer_sha256` doesn't match the live tokenizer's hash). On-disk token storage is **uint32** (not uint16) — uint16 silently wraps token ids ≥ 65536, and our vocab is 131,072.

**Training was native SentencePiece** ([`scripts/train_tokenizer_spm.py`](../../scripts/train_tokenizer_spm.py)), **not HF tokenizers**. HF tokenizers' Python BPE trainer hits a ~15 GB corpus ceiling — works for smoke but degrades token efficiency on real-scale training corpora. Production tokenizers should be trained on 100GB+ of representative data; SentencePiece native handles that.

**Required special tokens** (verified at training start): `BOS`, `EOS`, `PAD`, `UNK`. Loop refuses to start if any are missing or have id collisions.

---

## 6. What I'd flag as the genuine surprises / scars

These are the bugs that *had to be caught somewhere* — I'm listing them so you can judge whether the catches were good enough, and whether you'd expect more lurking ones.

1. **L3 canary caught a test-fixture bug, not a production bug** (today, 2026-05-12).
   - Symptom: ref and resumed runs had `step_match=True`, `data_position_match=True`, but `hash_match=False`.
   - Root cause: synthetic data iter used a stateful `np.random.default_rng(seed)`. On resume it was rebuilt fresh and yielded `batch[0..]` again, into step>0. So model trained on `[0,1,0,1]` instead of `[0,1,2,3]`.
   - Production path (`PackedCorpusReader`) was already resume-safe (peeks `data_position` from checkpoint manifest, seeks reader).
   - Fix: `batch[N]` now `SeedSequence([seed, N])` (per-step deterministic) + `start_step` parameter. `run_pretrain.py` synthetic branch peeks resume step via `find_resume_step`. Regression tests in [`tests/test_synthetic.py`](../../tests/test_synthetic.py) lock the invariant.
   - **Reviewer question**: do you trust that the production resume path is now correct, given we've only directly bitwise-verified the synthetic path? The packed-corpus seek is a different code path. Worth a separate L3 variant on a tiny real corpus?

2. **uint16 token storage would have silently corrupted everything** (caught pre-1B build).
   - Wraparound at 65536 in a 131072-vocab; first bug class that "trains fine" but produces gibberish on inference.
   - Fix: uint32 enforced at writer level + assert at reader level.

3. **mc4 multi-config collision in parallel runner** (caught during pilot build).
   - All 5 mc4 entries shared `dataset: mc4` → same output dir + log file → 4 processes crashed silently while 1 ran.
   - Fix: per-source-id output dir + per-source temp yaml in `scripts/run_parallel_builds.py`.

4. **pg19 over-fetch** (caught at smoke).
   - `tokenize_batch_size=1000` forced downloading 1000 books (~80M tokens) for a 2.5M target — 15 min stuck downloading a tiny slice.
   - Fix: char-aware early-flush in `_kept_doc_stream` — when `chars/3` estimated tokens would exceed target, flush early. 15 min → 1m08s.

5. **Build-monitor counter accumulation** (caught from a user screenshot showing 11/11 shards on a 1-shard build).
   - Monitor's stateful counters weren't reset between log re-parses → 11× accumulation per refresh.
   - Fix: reset counters at start of each `_parse_log_file` invocation. Plus multi-marker done detection (`packed_corpus_manifest_written` preferred over `build_one_source_done` due to stdout interleaving).

6. **`build_one_source_done` log corruption** (caught during pilot build).
   - structlog event interleaved with `print(json.dumps(summary))` on shared stdout → corrupted JSON on log readers.
   - Fix: detect done via `packed_corpus_manifest_written` event (on its own line, structlog-formatted).

7. **R2 region default 'nov' on a fresh pod** (caught on H200 today).
   - Cloudflare only accepts `wnam/enam/weur/eeur/apac/oc/auto`; pod's `AWS_DEFAULT_REGION` resolved to `'nov'`.
   - Workaround: `export AWS_DEFAULT_REGION=auto`. Not yet baked into repo.

---

## 7. Throughput & cost — the honest assessment

**Measured on 1×H200 SXM** (RunPod), pilot_250m, micro_batch=4, seq=4096, synthetic data:

```
tok/sec/device:    47,308
MFU:               3.4%
Peak HBM:          101.5 GB / 141 GB
1T @ 8×H200:       ~734 hr / ~30.6 days / ~$20.5K @ $3.5/hr/GPU
```

**Why MFU is 3.4% and why I think it's not a kernel bug**: H200 BF16 peak is 1979 TFLOPS. At 47K tok/sec × 6N FLOPs/token (1.5e9 for 250M) = 71 TFLOPS achieved. The 250M model has only ~1.5 GB of weights against a 141 GB GPU — kernel-launch + memory-bandwidth overhead dominate, not matmul. Typical MFU at 250M-scale on H100/H200 is in the 3-8% range; tensor cores saturate at 7B+.

**Bottleneck is the vocab projection**: `B × S × 131k` logits + same-size gradient buffer. At micro_batch=8 it tried to allocate 176 GB (won't fit in 141). XLA rematerialization only got it to 167 GB. We had to drop to micro_batch=4.

**Open improvement (not yet built)**: **Chunked cross-entropy** — split the 131k-row output projection into chunks. Drops peak HBM by ~3-5×, lets us run micro_batch=8 or 16. Should push to 70-100K tok/sec/device on H200; ~$20K → ~$10K for 1T. ~2-3 hr to build + test.

**Reviewer questions**:
- Is the chunked-CE work worth doing *before* the 1T base run, or just bank the 30 days?
- On 8×B200 (compute much higher, HBM 192 GB), do you expect the same micro_batch=4 ceiling or can we open up to 8?
- Have you seen MFU numbers from comparable from-scratch 1B runs in the last year? I want a reality check on whether 5-10% MFU at 1B is acceptable or whether we should be reaching for 15-20%.

---

## 8. Validations (what I have evidence for)

- **39 test files** under [`tests/`](../../tests), **~150+ test cases**. Key suites:
  - `tests/test_build_packed_corpus.py` — 27 tests including `test_huge_docs_dont_overfetch_with_target`
  - `tests/test_packed_corpus.py` — 56 tests including `TestR2StreamingMirror` class
  - `tests/test_synthetic.py` — 8 tests including the new resume-safety regression tests
  - `tests/test_canary.py`, `tests/test_canary_l3.py` — canary contract tests
- **9,751 LOC** in [`src/myllm/`](../../src/myllm)
- **Pilot v2 corpus** (40.7M tokens, 12 sources) — passes 5/5 L5 source-share-drift checks
- **End-to-end smoke pipeline** validated:
  - HF stream → filter chain → tokenize (Rayon batch) → packer (PackedCorpusWriter) → R2 multipart upload → delete local → manifest aggregation in-memory → corpus close

---

## 9. Open risks (what I'm still flying blind on)

Honest list — anything I haven't directly stress-tested.

1. **Multi-GPU FSDP**. Single-GPU validated; cross-device gradient reduction unverified. The classic BLOOM-tr11 bug class (tied-embedding gradient not reduced) is the kind L3-canary-multi-gpu would catch. We don't have it yet.
2. **Real-data resume**. L3 verifies synthetic. PackedCorpusReader seek logic is unit-tested but no end-to-end L3 on real corpus.
3. **Watchdog spike recovery in real training**. Unit-tested; never fired in a real run. If it triggers spuriously or fails to recover, it's a multi-day setback.
4. **Distillation teacher-cache infrastructure**. Top-K logits at corpus-position granularity — design exists ([`src/myllm/training/decay_phase.py`](../../src/myllm/training/decay_phase.py)), pre-compute pipeline not yet validated. If the cache is corrupted, the decay-phase distillation silently trains on garbage.
5. **WSM checkpoint averaging**. Code path exists ([`merge_checkpoints` in checkpoint.py:196](../../src/myllm/training/checkpoint.py#L196)); not used in a real run. The B1 audit found a namedtuple-restoration bug here (fixed); other paper-edge bugs may lurk.
6. **Decontamination index never built on real corpus**. The 11-benchmark live-build path is `~10-15 min` at run-start — fine for smoke. Pre-built index path exists but not exercised. For a 1T run we *must* pre-build.
7. **Tokenizer choice of 131K vocab is on the high side**. Most 1B-scale models use 32K-50K. Larger vocab gives better multilingual coverage + shorter sequences for non-English, but eats parameters (100M of our 250M is embeddings) and tanks MFU at training time via the output projection. **Reviewer question**: at our pilot/base sizes, what vocab would you have picked?
8. **No quarantine + replay validation in a real spike**. Quarantine writer exists; the replay path that re-introduces quarantined batches with a smaller LR has never been tested in anger.
9. **`scaled_init_for_residuals=False` at pilot** — the re-audit said enable for base. Not yet wired. Easy fix but needs a deliberate flag in `configs/base_1b.yaml`.
10. **Decontamination is disabled in the current 1B corpus build**. If we use this corpus for anything beyond Proxy A/B sanity, we need a re-build with decontam on, or live-decontam at training time (the loop supports it but it's slow).

---

## 10. Specific decisions where your judgment matters

Yes/no list. Each is something I have a leaning on but want a sanity check from someone who's done this before.

| # | Question | My leaning | Why I'd defer to you |
|---|---|---|---|
| 1 | Implement chunked-CE before 1T base? | Yes (cuts cost ~50%) | 2-3 hr build vs. $10K saved on 1T budget — but could be a distraction if 1T plan is risky for other reasons |
| 2 | Drop multilingual mix (12%)? | No — keep, English-only is a step back | mc4 fragility + Sangraha-Hindi-only smells like premature ambition |
| 3 | Run a 100B mid-scale before 1T? | Yes if cheap | Cost ~$2K, would shake out FSDP + real watchdog behavior + decontam-at-scale |
| 4 | Drop QK-norm and re-test muP transfer? | No (locked) | Pilot/base/wind-tunnel all agree per muP playbook; flipping now invalidates HP sweep |
| 5 | Switch tokenizer to 64K vocab? | Probably no | Already trained; switch costs 1-2 days. But MFU pain is real. |
| 6 | Pre-build decontam index for 1T? | Yes (mandatory) | Live-build at run-start = 10-15 min × 8 GPUs = wasted compute |
| 7 | Use 8-gram instead of 13-gram for decontam? | Uncertain | 8-gram (Llama 3 default) higher recall, more false positives; 13-gram tighter, may miss paraphrases |
| 8 | Bigger teacher cache (top-128 vs top-64)? | Stay at 64 | 2× storage; marginal student gain at our scale per literature |
| 9 | Start 1T at 8×B200 directly, or scale-up 1×B200 → 2× → 8×? | Scale-up | Wind-tunnel cheaper if FSDP has bugs; not yet validated |
| 10 | When to extend context 4096 → 8192? | End of pretrain, last 5-10% of tokens | Per Yarn / position-interp playbook; agree? |

---

## 11. What I'm asking you for

In rough priority:
1. **Architecture sanity** — pilot config + muP transfer plan. Anything you'd change before we run Proxy A?
2. **Throughput math** — does 30 days / $20K on 8×H200 for 1T at this MFU map onto your prior experience? Or are we leaving real performance on the table that you can spot?
3. **Open-risks gut check** — anything in §9 that you'd flag as "do not pass go without addressing"?
4. **Decision pings on §10** — even one-word verdicts on the yes/no list would unblock me.
5. **Smell-test on the corpus mix** — does the share table in §4 look like a defensible 1B-pretrain mix to you?

**Artifacts to open if you want to dig deeper**:
- [`docs/governance/model_card_v1.md`](../governance/model_card_v1.md) — canonical model spec
- [`docs/governance/data_card_v1.md`](../governance/data_card_v1.md) — full source dossier per dataset
- [`docs/governance/license_register.md`](../governance/license_register.md) — license review per source + per teacher
- [`configs/pilot_250m.yaml`](../../configs/pilot_250m.yaml) — pilot model config (annotated)
- [`configs/base_1b.yaml`](../../configs/base_1b.yaml) — base model config
- [`configs/data/pretrain_mix.yaml`](../../configs/data/pretrain_mix.yaml) — annotated source list with status notes
- [`scripts/canary_l3_resume.py`](../../scripts/canary_l3_resume.py) — bitwise-exact resume canary
- [`scripts/benchmark_throughput.py`](../../scripts/benchmark_throughput.py) — throughput + MFU
- [`artifacts/h200_throughput.json`](../../artifacts/h200_throughput.json) — measured throughput report

No prep needed from you in advance — read what catches your eye and ping me with reactions.

— harshit
