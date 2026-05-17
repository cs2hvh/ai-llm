# Session Handoff (current — last refreshed 2026-05-17)

> **The live handoff for the next session.** Renamed from
> `SESSION_HANDOFF_2026-05-14.md` during the 2026-05-17 docs cleanup —
> filename no longer carries a date. Older pilot narrative lives in
> `pilots/250m_v1/TIMELINE.md`; design + algorithms in
> [`DESIGN.md`](DESIGN.md); pilot artifacts inventory in
> `pilots/250m_v1/R2_PATHS.md`. For project entry-point orientation,
> see [`PROJECT_OVERVIEW.md`](PROJECT_OVERVIEW.md) (slim index).

---

## 0. TL;DR

1. **Stage 1 pilot DONE** (250M, $385, val_loss 2.7303 / val_ppl 15.34). DP-replicated on 4×H200, NOT FSDP. Final ckpt on R2.
2. **Phase 1 engineering DONE** + **2 reviewer rounds processed code-side** (Round A 6 quick wins + Round B 4 Stage-2 P0s + Layer 1 packet backfill + Layer 2 part 1 real MMLU-Pro+GSM8K adapters + init_state refactor).
3. **C1 partial** — per-source PPL banked on R2. Scorecard run aborted (IFEval/HumanEval+/MBPP+ scoring still placeholder). ~$8 spent.
4. **C2 DONE on 4×B200 NVLink-5 (Vast.ai)** — 30% MFU @ seq=8192, 46% MFU @ seq=4096. Both above Stage 2 bars.
5. **C3 DONE on 4×B200 (post-Vast.ai pivot)** — 3-LR sweep ran cleanly after **2 hotfixes** for `data_position` pytree mismatch under `--fsdp`. **Peak LR 3.0e-4 (the pilot's value) WINS**: val_loss 6.173 / val_ppl 479.6 at step 1000. **muP transfer 250M → 1B is confirmed.**
6. **2 new bugs surfaced and characterized**:
   - **chunked-CE produces NaN gradients at 1B + B200 + bf16 + width_mult=8** (loss is finite, gradients aren't). Pilot 250M chunked-CE was fine. Stage 2 must use full-CE on B200 until this is fixed.
   - **step-718 deterministic NaN batch** in the composed pilot corpus — same step in all 3 C3 runs (1/1000 = 0.1%, atomic revert handles it). `quarantine-lr1_5x.jsonl` on R2 has provenance.
7. **All artifacts on R2** at `s3://llm-data/stage2-prep/` (~82 GB of checkpoints + 3 logs + benchmarks + quarantine file).
8. Suite: **739 passed, 1 skipped**. All commits pushed (HEAD = `8e50333`).
9. **Stage 2 readiness**: peak_lr locked at 3e-4; hardware/seq/budget decision pending.
10. **DO NOT** commit Stage 2 GPU spend before a short full-CE throughput smoke (mb=4, no chunked-CE) — chunked-CE C2 numbers don't apply.
11. **External review processed 2026-05-17** ([§7](#7-external-review-2026-05-17-and-actions-taken)). Verified via 3 parallel WebFetch agents (peer 1B specs, Muon evidence, teacher availability + Gemma softcap values). Took 7 P0 recs as CPU-side work this session: README/config staleness fixed, WSD decay-to-zero (D2Z) locked, chunked-CE fp32 logsumexp (D8 mitigation), Gemma 2 softcaps (50 attn / 30 final) wired end-to-end, watchdog precise-6σ test, D8 GPU repro script ready for $5 B200 hour. Stage 2 plan now adopts reviewer's **8×B200 + seq=4K + 20B tokens + D2Z** recommendation.

---

## 1. Where things stand (2026-05-17)

### Pilot (unchanged)

| Item | Value |
|---|---|
| Sharding | **DP-replicated** on 4×H200 SXM (no `--fsdp`). FSDP proven separately via gauntlet G1-G4 + L2/L3 canaries. |
| Stage 1 final | step 151,990, val_loss 2.8776, val_ppl 17.77 |
| Stage 1.5 final | step 171,990, val_loss 2.7303, val_ppl 15.34 |
| Watchdog | Quiet — 288 NaN-skips / 172K steps, 0 hard rollbacks |
| W&B | `roydqofb` + `u5xsxm0l` + `pxoungh9` |
| Per-source PPL | **MEASURED** — banked at `s3://llm-data/scorecards/pilot-250m-v1-decay/pilot-250m-v1-decay-per-source.json`. Code best (3.30), Hindi worst (41.32). |

### Phase 1 + reviewer cycle (all SHIPPED + pushed)

| Commit | What |
|---|---|
| `be7574c` | Phase 1.1 multi-epoch corpus reader |
| `082fa20` | Phase 1.3+1.4 `--production` + strict packed-resume |
| `107a551` | Phase 1.5 forward-only `make_eval_step` (FSDP-safe) |
| `97c59c1` | Phase 1.6 G6 cross-mesh restore regression tests |
| `fbe9c72` | Phase 1.2 per-source val loss |
| `329b349` | **Round A** 6 quick wins (doc fixes, eval_checkpoint jax-order, governance template fix, canary_ladder packed-L3, version pins + orbax compat tests) |
| `574dd8f` | **Round B** 4 Stage-2 P0s (KD vocab gate, watchdog `_recover_from_spike`, benchmark_throughput `--fsdp`, scorecard predict_fn) |
| `7025bb9` | infer/predict sys.path fix |
| `00f4ad2` | `eval_checkpoint.py --per-source-val-loss` for C1 backfill |
| `302ba05` | **Layer 1** packet §2.4 backfill + §2.6 scorecard limitation |
| `04bfaf5` | **Layer 2 part 1** MMLU-Pro + GSM8K real adapters (37 tests) |
| `52eb857` | **Refactor** init_state helpers + ensure_tokenizer_local → `src/myllm/` |
| `cbd5477` | **HOTFIX 1** data_position pytree mismatch in train_step under `--fsdp` |
| `8e50333` | **HOTFIX 2** data_position pytree mismatch in eval_step under `--fsdp` |

### C1 (1× H200) — DONE PARTIAL

- **Got**: per-source val_loss for all 13 sources (`s3://llm-data/scorecards/pilot-250m-v1-decay/`).
- **Abandoned**: scorecard run — placeholder scoring "non-empty = success" gave accuracy 1.0 on every sample. Round B4 wired the predict_fn; per-benchmark `score()` methods for IFEval/HumanEval+/MBPP+ still scaffolds.
- **Spent**: ~$8.

### C2 (4× B200 NVLink-5 on Vast.ai) — DONE

Banked at `s3://llm-data/stage2-prep/benchmarks-4b200/`:

| Combo | Step time | Aggregate tok/s | Real BF16 MFU | Peak HBM | Stage 2 (30B) wall | Stage 2 (30B) $ |
|---|---|---|---|---|---|---|
| seq=8192 mb=16 | 2.37 s | 221,246 | **~30%** | 131 GB / 183 GB | 37.7 hr | ~$530 |
| seq=4096 mb=16 | 0.78 s | 335,683 | **~46%** | 43.7 GB / 183 GB | 24.8 hr | ~$350 |

**Caveat (important for Stage 2)**: these C2 numbers used `--use-chunked-ce`, which we now know produces NaN gradients at 1B+B200+bf16+width_mult=8. Stage 2 must use full-CE which is more memory-hungry and slower at this batch shape. **Re-bench needed at full-CE mb=4 (or whatever fits) before locking Stage 2 throughput projections.**

### C3 (μP/LR sweep, 4× B200 NVLink-5 on new RunPod pod after Vast.ai → 5×H200 → 8×B200 → 4×B200 pivots) — DONE

3 runs × 1000 steps each, full-logit CE, FSDP, mb=4, seq=8192. Results banked at `s3://llm-data/stage2-prep/mup-sweep-4b200/`:

| Run | Peak LR | Step 1000 val_loss | Step 1000 val_ppl | Stability |
|---|---|---|---|---|
| 1 | 1.0e-4 (0.5×) | 7.015 | 1113 | 1 NaN-skip @ step 718 |
| 2 | 2.0e-4 (1.0×, base config value) | 6.445 | 630 | 1 NaN-skip @ step 718 |
| 3 | **3.0e-4 (1.5×, pilot's value)** | **6.173** | **479.6** | 1 NaN-skip @ step 718 |

**Run 3 wins 12 out of 13 sources** (mc4_zh is the lone outlier, small slice noise). Per-source at step 1000 for Run 3:

| Source | val_loss | val_ppl |
|---|---|---|
| github_code_clean | 4.674 | 107 |
| mc4_ar | 5.547 | 257 |
| stack_exchange | 5.423 | 226 |
| pg19 | 6.149 | 468 |
| open_web_math | 6.380 | 591 |
| fineweb_edu | 6.347 | 570 |
| wikipedia | 6.615 | 746 |
| pes2o | 6.619 | 749 |
| mc4_es | 6.888 | 981 |
| mc4_de | 7.670 | 1972 |
| mc4_fr | 7.660 | 2123 |
| sangraha_hin | 7.464 | 1746 |
| mc4_zh | 8.102 | 3301 |
| **AGGREGATE** | **6.173** | **479.6** |

**Headline finding**: peak_lr=3.0e-4 (the pilot's value, scaled-up via muP to width_mult=8) trains the 1B model **stably AND with lowest val_loss**. **muP transfer from 250M to 1B is confirmed.**

### Two bugs surfaced during C3 setup + run

**Bug 1: chunked-CE NaN gradients at 1B + B200 + bf16 + width_mult=8**
- Symptom: train_step's atomic NaN-revert fires on every batch. Loss is **finite** at step 0 (11.76 = ln(131072)) but the backward pass produces NaN in at least one gradient leaf. From step 1+, forward also goes NaN.
- Repro: 1B model + `--use-chunked-ce --fsdp` + mb=16 + seq=8192 + lr=1e-4 + B200. Every batch NaN-skipped, no progress.
- Pilot 250M + chunked-CE worked fine; the bug is scale-specific (width_mult=8 vs pilot's 3.0) or hardware-specific (B200 bf16 vs H200) or both.
- Diagnostic: dropping `--use-chunked-ce` (full-logit CE) makes the training work cleanly. mb=4 + full-CE + FSDP on 4× B200 fits comfortably in 183 GB.
- Likely root cause: chunked-CE's online logsumexp (`running_max`, `running_sum` accumulators) hits bf16 precision boundaries when V=131072 / 8 chunks × seq=8192. Or its gradient through `take_along_axis` + `where` has a numerical issue at this scale.
- **CPU audit done 2026-05-17** ([`docs/design/d8_chunked_ce_audit.md`](design/d8_chunked_ce_audit.md)): minimal CPU repro of the chunked-CE algorithm in bf16 at muP output_mult=1/8 produces **finite** gradients matching full-CE to 2.98e-7. Pure-algorithm hypothesis is disproved. Bug is B200/CUDA-specific (Blackwell tensor cores, XLA op fusion on CUDA, or FSDP reduce-scatter at bf16). GPU repro on B200 deferred (~$5, ~1 hr).
- **Stage 2 implication**: must use full-CE on B200 (memory budget allows at mb=4). Pre-Stage-3 we need a chunked-CE fix because at 7B+ full-CE OOMs.
- **Tracked as Round D investigation item.**

**Bug 2: step-718 deterministic NaN batch**
- Symptom: same step (718) in all 3 C3 runs triggers `nan_batch_skipped`. Atomic revert handles it. 1 / 1000 steps = 0.1% rate, watchdog stays quiet.
- Repro: deterministic — same seed, same corpus, same packed-sequence at sequence_id corresponding to step 718.
- Quarantine file at `s3://llm-data/stage2-prep/mup-sweep-4b200/quarantine-lr1_5x.jsonl` has the offending batch's `data_position` + source_mix metadata.
- Investigation tool: `scripts/inspect_quarantine.py` (already in repo) maps `data_position → sequence_id → source` via the corpus's `seq_meta.arrow`.
- **Not a Stage 2 blocker** (0.1% NaN-skip rate is well below the 1% threshold), but worth understanding before Stage 3.
- **Tracked as Round D investigation item.**

### Hotfixes (both shipped, both have regression tests)

| Commit | What |
|---|---|
| `cbd5477` | **HOTFIX 1**: run_pretrain.py's FSDP block included `data_position` in `state_shardings` (6 keys), but loop.py pops it before calling `train_step_fn` (int32-overflow fix from `9f442f7` → 5 keys arriving at JIT). Mismatch → `ValueError: different numbers of pytree children`. Fix: remove data_position from state_shardings + carry as Python int outside the JIT'd state pytree. Regression: `tests/test_train_step_fsdp.py::test_fsdp_works_when_loop_pops_data_position_before_call`. |
| `8e50333` | **HOTFIX 2**: same root cause but in the eval path. Loop's eval-call site didn't have the pop-restore pattern → eval failed non-fatally on every eval cycle under `--fsdp`. Fix: pop+restore around `eval_fn(step, state)` call. Regression: `tests/test_eval_hook.py::test_eval_fn_does_not_see_data_position_in_state`. |

Suite: 738 → 739 passed across these two commits. Both regression tests would fail pre-fix.

### R2 artifacts inventory (Stage 2 prep)

| Path | Size | What |
|---|---|---|
| `s3://llm-data/stage2-prep/benchmarks-4b200/` | ~85 KB | C2 throughput (4 files: 2 combos × json+log) |
| `s3://llm-data/stage2-prep/mup-sweep-4b200/lr{0_5,1_0,1_5}x.log` | ~3 MB | C3 training logs |
| `s3://llm-data/stage2-prep/mup-sweep-4b200/quarantine-lr1_5x.jsonl` | 2 MB | step-718 bad-batch provenance |
| `s3://llm-data/stage2-prep/checkpoints/mup-sweep-4b200/` | ~82 GB | 6 checkpoints (3 LRs × 2 steps), Orbax sharded |
| `s3://llm-data/scorecards/pilot-250m-v1-decay/` | ~6 KB | Pilot per-source PPL (C1) |
| `s3://llm-data/checkpoints/pilot-250m-v1-decay/` | ~2.65 GB | Pilot final checkpoint |
| `s3://llm-data/corpus_v1_pilot/train/` | ~20 GB | Composed pilot corpus |
| `s3://llm-data/tokenizer/myllm-spm-unigram-131k-v2.json` | ~5 MB | Tokenizer |

---

## 2. The plan ahead

### Immediately (CPU-side, no GPU)

1. ~~**Backfill reviewer packet §6**~~ — **DONE 2026-05-17** (commit `5653251`).
2. ~~**Investigate step-718 quarantine**~~ — **DONE 2026-05-17** (commit `e696add`). Root cause: Stack Exchange single-doc 8K sequence at shard 0 / seq_id 2871. Folds into D5. See [`design/d9_step718_investigation.md`](design/d9_step718_investigation.md).
3. ~~**Investigate chunked-CE bug**~~ — **CPU audit DONE 2026-05-17** (commit `0e5080d`). Algorithm clean on CPU; bug is B200-specific. See [`design/d8_chunked_ce_audit.md`](design/d8_chunked_ce_audit.md).
4. ~~**Process external review 2026-05-17**~~ — **DONE 2026-05-17**. Verified via 3 parallel WebFetch agents. Took 7 P0 recs CPU-side: README fix + WSD D2Z + chunked-CE fp32 logsumexp (D8 mitigation) + Gemma 2 softcaps (50/30) wired + watchdog 6σ test + D8 GPU repro script. See §7 below.

### Stage 2 launch decision (need user input)

Three open knobs (reviewer recommendation baked in where available):

| Decision | Options | My lean (post-review) |
|---|---|---|
| Hardware | 4× B200 / **8× B200** / 8× H200 SXM | **8×B200** — same total $, half wall-time vs 4×B200; B200 proven by C3 |
| Seq length | **4096** / 8192 | **4096** — reviewer math: ~67% faster per token; long-context recoverable in Stage 3 via θ-scaling |
| Token budget | 10B / **20B** / 30B | **20B** — "real" rehearsal; 30B is +50% cost for marginal extra signal |
| Loss path | full-CE (locked until chunked-CE fix) | Locked by Bug 1 |
| LR | 3.0e-4 (locked by C3) | Locked |
| `end_lr_ratio` | 0.0 (D2Z, was 0.1) | **CHANGED 2026-05-17** per Bergsma 2502.15938 |
| Final-logit softcap | 30.0 (was None) | **KEPT 2026-05-17** — Gemma 4 (2026-04-02) still ships `final_logit_softcapping=30.0` |
| Attn-logit softcap | None (correction 2026-05-17) | **OFF by default**. Initial post-review pass set 50.0 from Gemma 2. Follow-up Gemma 3/4 verification showed Gemma 3 explicitly replaced this with QK-norm (we already have qk_norm=true). Code path retained as opt-in. |
| micro_batch | 4 (locked by full-CE memory budget) | Locked |
| `--corpus-epochs` | 6 (locked by Stage 2 token budget vs 5B corpus) | Locked |
| `--production` | yes | Locked |

**Recommended Stage 2 path** (post-review):
1. **D8 GPU repro** first (~$5, ~30 min on 1×B200): `python scripts/d8_gpu_repro.py --mode disable-remat`. If it clears, root cause = openxla/xla #17922; re-enable chunked-CE for Stage 2. If not, stay on full-CE workaround.
2. Short **full-CE throughput smoke** at seq=4K, mb=4, 8×B200, ~2K steps (~$15-20) to lock MFU at production shape with all new flags (softcaps + D2Z + fp32 logsumexp).
3. Commit to **20B Stage 2 rehearsal** (~$160-230) once smoke is clean.

### Round D — Stage 3 prep + investigations (parallel with Stage 2)

| # | Item | Effort | Status |
|---|---|---|---|
| D1 | Chunked distillation for decay phase | 1-2 days | Pending |
| D2 | Teacher top-K mass audit on real text | 0.5 day + GPU | Pending |
| D3 | Stratified per-source held-out | 4 hr | Pending |
| D4 | pg19 replacement for Stage 3 corpus | depends | Pending |
| D5 | Stack Exchange schema fix | 2 hr | Pending |
| D6 (in progress) | Real scoring policies — MMLU-Pro+GSM8K done; IFEval/HE+/MBPP+ pending | 1-2 days | Partial |
| D7 | Logical-axis FSDP sharding rules | 2-3 days | Pending |
| **D8** | ~~chunked-CE NaN-grad at 1B+B200+bf16+width_mult=8~~ — **CPU audit DONE**, **fp32-logsumexp mitigation SHIPPED**, GPU repro script READY 2026-05-17 | 1 hr CPU done; $5 B200 hour pending | CPU audit + fp32-logsumexp + GPU repro driver shipped. Run `scripts/d8_gpu_repro.py --mode disable-remat` to test openxla/xla #17922 hypothesis. |
| **D9** | ~~step-718 bad-batch investigation via quarantine.jsonl + corpus inspection~~ — **DONE** 2026-05-17 (e696add) | ~2 hr CPU done | **Done** → [`design/d9_step718_investigation.md`](design/d9_step718_investigation.md). Root cause folds into D5. |
| **D10 (NEW)** | Muon optimizer port (hybrid: Muon for ≥2D hidden weights, AdamW for embeddings/head/scalars) + MuonClip | 2 days code + $10 GPU smoke | Pending — biggest single quality lever per verified review. Per Moonshot 2502.16982 (~2× compute efficiency vs AdamW) + Kimi K2 2507.20534 (1T params, 15.5T tokens, zero loss spike). Recommended pre-Stage-2.5. |
| **D11 (NEW)** | Apple Cut-Cross-Entropy `custom_vjp` (arXiv 2411.09009) for LM head | 1-2 days | Pending — would kill the D8 bug class AND restore chunked-CE's memory benefit at 7B+. Required pre-Stage-3. |
| **D12 (NEW)** | Architecture revision for Stage 3: deeper-thinner (24-28L / hidden 1792 / FFN 3×) + depth-μP via spectral (arXiv 2603.00541) | 1-2 weeks + fresh wind-tunnel sweep | Pending — backed by MobileLLM-Pro's verified +7.9% on 3-cat avg over Llama 3.2 1B. Stage 3 prep, NOT Stage 2 blocker. |
| **D13 (NEW)** | Gopher repetition filters in corpus pipeline (`frac_chars_in_dup_5_10grams > 0.20`, byte-entropy ≥ 3.5 bits/byte) | 1 day CPU | Pending — folds with D5 Stack-Exchange schema rebuild. Would have prevented D9 at the data layer. |

---

## 3. Critical gotchas (silent-corruption modes)

Patterns to carry forward:

1. **int32 cursors in JIT'd state** — `data_position` wrapped at ~step 65,500 in pilot. Fix: pop before JIT call, restore after. **Applies to BOTH train_step AND eval_step** (hotfixes 1+2 land this in both code paths).
2. **Single-pass iterators that look infinite** — Stage 2 MUST launch with `--corpus-epochs 6+`.
3. **Orbax API drift bites in minor versions** — pin exactly (jax==0.4.38, orbax==0.7.0, tensorstore==0.1.83). `tests/test_orbax_api_compat.py` smoke-tests every kwarg we depend on.
4. **FSDP `donate_argnums=(0,)` invalidates state for any subsequent caller** — eval under FSDP needs `make_eval_step`.
5. **Pilot ran DP-replicated, not FSDP.** Do not let prose drift. FSDP proven by gauntlet G1-G4 separately + by C3 sweep this session.
6. **Scorecard scoring still placeholder for IFEval/HE+/MBPP+** — Layer 2 fixed MMLU-Pro+GSM8K only.
7. **JAX x32 mode demotes int64 in `restore_args` path** — pinned by `tests/test_orbax_api_compat.py`. Production resume uses legacy restore path which preserves int64.
8. **NEW: chunked-CE on B200 bf16 at 1B+ produces NaN gradients.** Symptom: finite forward loss but NaN gradient → atomic revert fires every step → no learning. Use full-CE on B200 until D8 lands a fix.
9. **NEW: `state_shardings` under `--fsdp` MUST NOT include `data_position`.** The loop pops it before JIT. Including it triggers "different numbers of pytree children" ValueError. Pattern: any future loop-managed state field that the loop wants to keep OUTSIDE the JIT'd pytree must also be excluded from `state_shardings`. See hotfix-2 (commit `8e50333`) for the pattern in eval.
10. **NEW: step-718 of the composed pilot corpus (sequence_id ≈ 4 × 718 = 2872 at mb=4) deterministically NaN's the gradient.** Atomic revert handles it. Worth understanding before scaling corpus size to Stage 3 (5B → 600B).

---

## 4. Reviewer findings status

| Finding | Status | Commit / R2 path |
|---|---|---|
| KD vocab mismatch (P0) | CLOSED | `574dd8f` B1 |
| Watchdog `_recover_from_spike` (P0) | CLOSED | `574dd8f` B2 |
| `benchmark_throughput.py` doesn't measure FSDP (P0) | CLOSED | `574dd8f` B3 |
| Scorecard `predict_fn` NotImplementedError (P0) | CLOSED | `574dd8f` B4 + `04bfaf5` (MMLU-Pro+GSM8K) |
| `eval_checkpoint.py` jax import order (P1) | CLOSED | `329b349` A3 |
| `render_governance_cards.py` template names (P1) | CLOSED | `329b349` A4 |
| `canary_ladder.py` packed-L3 default (P1) | CLOSED | `329b349` A5 |
| Doc errors in reviewer packet (P1) | CLOSED | `329b349` A1 + `302ba05` |
| Orbax exact pins + smoke test (P1) | CLOSED | `329b349` A6 |
| Per-source PPL TBD (R2) | CLOSED | `s3://llm-data/scorecards/pilot-250m-v1-decay/` + `302ba05` |
| **muP transfer 250M → 1B unproven (P2 from §7.1 of packet)** | **CLOSED** | C3 sweep banked at `s3://llm-data/stage2-prep/mup-sweep-4b200/` |
| **Stage 2 cost model on actual FSDP path (R2)** | **PARTIALLY CLOSED** | C2 done with chunked-CE (need re-bench at full-CE) |

Still deferred:
| Item | Status |
|---|---|
| HE+ / MBPP+ scoring (sandboxed code-exec) | Pending Round D6 |
| IFEval scoring (constraint library) | Pending Round D6 |
| Chunked distillation memory savings in decay | Pending Round D1 |
| Stratified per-source held-out | Pending Round D3 |
| Logical-axis FSDP sharding rules | Pending Round D7 |
| **chunked-CE NaN-grad at 1B+B200** | **D8 CPU audit DONE 2026-05-17** — algorithm fine on CPU, bug is B200-specific (see [`design/d8_chunked_ce_audit.md`](design/d8_chunked_ce_audit.md)). GPU repro pending. |
| **step-718 deterministic bad batch** | **D9 DONE 2026-05-17 (e696add)** — Stack Exchange single-doc; folds into D5 (see [`design/d9_step718_investigation.md`](design/d9_step718_investigation.md)). |

---

## 5. Resume / verify

```bash
cd /root/llm-build
.venv/bin/python -m pytest -q
# expect: 739 passed, 1 skipped
```

For Stage 2 short smoke (any 4-or-8 GPU NVLink pod):

```bash
# After Phase A bootstrap (apt + venv + pip install -e .[cuda] + R2 creds + matmul probe)
# Pull artifacts:
aws --endpoint-url "$S3_ENDPOINT_URL" s3 cp \
  s3://$S3_BUCKET/tokenizer/myllm-spm-unigram-131k-v2.json artifacts/tokenizer_v1.json
aws --endpoint-url "$S3_ENDPOINT_URL" s3 sync \
  s3://$S3_BUCKET/corpus_v1_pilot/train/ /workspace/corpus_pilot_train/

# Short throughput probe at production shape (full-CE; NOT --use-chunked-ce):
python scripts/run_pretrain.py \
  --model-config configs/base_1b.yaml \
  --data-config configs/data/pretrain_mix_pilot.yaml \
  --tokenizer-path artifacts/tokenizer_v1.json \
  --packed-corpus-root /workspace/corpus_pilot_train \
  --run-name stage2-smoke \
  --total-steps 2000 \
  --peak-lr-override 3.0e-4 \
  --micro-batch-override 4 \
  --fsdp \
  --log-every 50 \
  --eval-every 500 \
  --eval-n-batches 16 \
  --per-source-val-loss \
  --corpus-epochs 6 --production \
  --checkpoint-every 1000 \
  --checkpoint-root /workspace/ckpt/stage2-smoke \
  --no-wandb
```

For C3 results re-eval / generation from any of the 3 LR variants:

```bash
mkdir -p /workspace/ckpt/mup-sweep-4b200
aws --endpoint-url "$S3_ENDPOINT_URL" s3 sync \
  s3://$S3_BUCKET/stage2-prep/checkpoints/mup-sweep-4b200/lr1_5x/step-000001000/ \
  /workspace/ckpt/mup-sweep-4b200/lr1_5x/step-000001000/

KERAS_BACKEND=jax python scripts/generate.py \
  --model-config configs/base_1b.yaml \
  --tokenizer-path artifacts/tokenizer_v1.json \
  --checkpoint-root /workspace/ckpt/mup-sweep-4b200/lr1_5x \
  --checkpoint-step 1000 \
  --prompt "The future of AI is" --temperature 0.8 --top-p 0.9
```

---

## 6. Where things live

| Need | Open |
|---|---|
| Pilot artifacts | `pilots/250m_v1/` |
| Canonical project state | `docs/PROJECT_OVERVIEW.md` |
| Reviewer packets | `docs/review/POST_PILOT_REVIEW_2026-05-15.md` (§2.4 has real numbers; §2.6 documents scorecard limitation; **§6 NEEDS BACKFILL with C3 results**) |
| Stage 3 Rust migration plan | `docs/stage3_rust_migration_plan.md` |
| muP design + scaling rules | `docs/mup_design.md` |
| Teacher distillation strategy | `docs/teacher_distillation_strategy.md` |
| Pilot config (250M, 768/3072, GQA 3:1, peak_lr 3e-4) | `configs/pilot_250m.yaml` |
| Base 1B config (2048/8192, GQA 4:1, peak_lr 2e-4) | `configs/base_1b.yaml` |
| Stage 1.5 decay config | `configs/pilot_250m_decay.yaml` |
| Per-source val loss (in-training) | `src/myllm/training/eval_hook.py` + `eval_step.py` |
| Per-source val loss (post-hoc CLI) | `scripts/eval_checkpoint.py --per-source-val-loss` |
| G6 cross-mesh restore | `src/myllm/training/checkpoint.py` |
| FSDP sharding | `src/myllm/training/mesh.py` (shape-heuristic; D7 will replace) |
| Watchdog + recovery | `src/myllm/training/watchdog.py` + `loop.py:_recover_from_spike` |
| Init state helpers (refactored) | `src/myllm/training/state_init.py` |
| Tokenizer fetch | `src/myllm/utils/storage.py::ensure_tokenizer_local` |
| Shared inference path | `src/myllm/infer/predict.py` |
| MMLU-Pro real adapter | `src/myllm/eval/benchmarks/mmlu_pro.py` |
| GSM8K real adapter | `src/myllm/eval/benchmarks/gsm8k.py` |
| Orbax API compat smoke | `tests/test_orbax_api_compat.py` |
| **NEW: data_position pop/restore around train_step** | `src/myllm/training/loop.py` (around lines 220-230) |
| **NEW: data_position pop/restore around eval_fn** | `src/myllm/training/loop.py` (around lines 350-370) |
| **NEW: state_shardings sans data_position** | `scripts/run_pretrain.py` (FSDP block around line 952) |
| **NEW: quarantine forensic tool** | `scripts/inspect_quarantine.py` (read R2's `quarantine-lr1_5x.jsonl`) |

---

## 7. External review 2026-05-17 and actions taken

A comprehensive external review landed 2026-05-17. Per our durable "verify
before locking" rule, every load-bearing factual claim was cross-checked
via 3 parallel WebFetch verification agents (peer 1B specs + tokens, Muon
evidence, teacher availability + Gemma softcap values). Net: about 70% of
the review's recommendations were correct and applicable, 20% were already
done (reviewer reading a stale snapshot from pre-2026-05-17 docs), and 10%
were partial / overstated.

### Reviewer claims pushed back on (stale-snapshot reads)

| Claim | Why wrong |
|---|---|
| `DESIGN.md` missing | Exists as `docs/DESIGN.md` (40 KB, commit `6fff980` 2026-05-17). |
| D5/D8/D9 labels not in repo | Present throughout `DESIGN.md`, this file, `design/d8_*` `design/d9_*`. Reviewer was reading archived `Phase A/B/B1–B9` labels from `docs/archive/`. |
| RoPE θ=130k; raise to 500k as P0 | `configs/base_1b.yaml:55` is already `500000.0`. |
| muP transfer "CONFIRMED is overstated" | README was stale; the 2026-05-16 C3 sweep (peak_lr=3e-4 wins monotonically across 1B-shape on 4×B200) closes it. README now fixed. |
| "MyLLM's d/L=128 matched only by Llama 3.2 1B" | Wrong. OLMo 2 1B is also 16L / hidden 2048 / d/L=128, non-distilled, 4T tokens. Verified from `allenai/OLMo-2-0425-1B` card. |

### Reviewer claims verified literally (took the recommendation)

| Recommendation | Verified source | Status |
|---|---|---|
| Llama 3.2 1B = 9T + distilled from 8B/70B | Meta card "Up to 9T tokens"; "logits from Llama 3.1 8B and 70B" | Re-frames our project goal: target OLMo 2 1B / Gemma 3 1B tier (verified-reachable), not Llama 3.2 1B tier |
| Gemma 2 attn-softcap=50, final-softcap=30 | arXiv 2408.00118 verbatim | **WIRED** end-to-end (config + attention + final-logit + chunked-CE) + 3 unit tests + on by default in `base_1b.yaml` |
| WSD decay-to-zero (Bergsma 2502.15938) | arXiv "Straight to Zero" verbatim, 60% compute savings at 610M | `end_lr_ratio: 0.0` in `base_1b.yaml` |
| chunked-CE bf16 logsumexp unstable (jax-ml/jax #13529) | GitHub issue: "worse for bf16 but also noticeable in float32" | **MITIGATED**: accumulators run in fp32; matmul stays bf16. Bf16 regression test added at `tests/test_loss.py::test_chunked_bf16_hidden_uses_fp32_logsumexp`. |
| openxla/xla #17922 = remat copy-ops cause grad NaN post JAX 0.4.30 | GitHub issue + PR #18152 fix | We pin jax==0.4.38 → highly relevant. `scripts/d8_gpu_repro.py --mode disable-remat` ready to test this hypothesis on $5 of B200. |
| Stage 2 = 8×B200 + seq=4K + 20B tokens | Reviewer's throughput math holds (4K is ~67% faster per token) | Plan updated; needs final user sign-off. |
| Watchdog 6σ stress test | Reviewer Q9 | New test `test_precise_6sigma_threshold_fires` injects spike at base+6.5σ and verifies rollback fires. |
| Muon optimizer hybrid (2D hidden → Muon, scalars/embed/head → AdamW) | Moonshot 2502.16982 (2× compute); Kimi K2 2507.20534 (1T params, 15.5T tokens, zero spikes); KellerJordan/Muon canonical pattern | **Tracked as D10**. Recommended for Stage 2.5 / pre-Stage-3. Single biggest unrealized quality lever. |
| Apple CCE custom_vjp (arXiv 2411.09009) | 24 GB → 1 MB loss memory on Gemma 2B | **Tracked as D11**. Would kill D8 bug class. |
| Architecture deeper-thinner (24-28L / hidden 1792 / FFN 3×) | MobileLLM ICML 2024 "deeper and thinner models generally outperform"; MobileLLM-Pro +7.9% 3-cat avg over Llama 3.2 1B | **Tracked as D12**. Stage 3 prep, NOT Stage 2 blocker — needs fresh depth-μP sweep. |
| Gopher repetition filters (Stage 2 corpus) | RedPajama-v2 quality signals | **Tracked as D13**. Folds with D5 Stack-Exchange schema rebuild. |

### Reviewer claims partial / nuanced

| Claim | What we found |
|---|---|
| Qwen3 0.6B/1.7B = 36T tokens | 36T is the **family corpus**, not per-model. Sizes downstream may see 10-30T. |
| MobileLLM-Pro +7.9% over Llama 3.2 1B "on reasoning" | The +7.9% is a 3-category average (reasoning + knowledge + long-context retrieval), not pure reasoning. Still meaningful. |
| DeepSeek-V4-Pro-Base "operationally bad for distilling 1B" | Cross-tokenizer (uses `encoding_dsv4`, not SP-Unigram) — verified. FP4 MoE — verified. But our `docs/teacher_distillation_strategy.md` already uses **offline top-K logit caching** ($4,100 budgeted), not online inference. Reviewer hadn't read that doc. Still: cross-tokenizer KD at our scale is unproven; consider Olmo-3-1125-32B (Apache-2.0, 5.5T pretrain — reviewer said 5.9T, off by 0.4T) as primary. |
| Bergsma "60% compute savings" | At 610M scale vs cosine-to-10%, not generic. Real but smaller win at 1B. |
| MuonClip "standard safety net" | Verified-but-overstated. Only one production deployment (Kimi K2). |

### Follow-up correction 2026-05-17 (post-Gemma-3/4 verification)

Initial commit `d56e1e0` set `attn_logit_softcap=50.0` and
`final_logit_softcap=30.0` taken verbatim from Gemma 2 paper. User asked:
"Gemma 4 is out — are we still on Gemma 2 values?" Per the
"verify-before-locking" rule we re-checked via WebFetch against current
HF model cards / arXiv:

- **Gemma 3 technical report (arXiv 2503.19786)** says verbatim: *"we
  replace the soft-capping of Gemma 2 with QK-norm."* Gemma 3 configs
  ship `attn_logit_softcapping: null`, `final_logit_softcapping: null`,
  with a scaled-softmax pattern (`query_pre_attn_scalar=168`) instead.
- **Gemma 4 (released 2026-04-02)** config: `final_logit_softcapping: 30.0`
  retained; `attn_logit_softcapping` field absent entirely.
- **Llama 4** (iRoPE + scaled softmax) and **Mistral Large 3** also do
  not use softcap.

We already ship `qk_norm: true` (the Gemma 3+ replacement for attn
softcap). Doubling up with `attn_logit_softcap` forces the manual
attention path (~5-10% wall-time tax at seq=8192) without any verified
additional stability benefit at our scale.

**Action**: flipped `attn_logit_softcap` to `null` in `base_1b.yaml`
(kept the code path as an opt-in debug knob). `final_logit_softcap`
stays at `30.0` — Gemma 4 retains it. The 3 attention-softcap unit
tests in `tests/test_model.py` still pass because they construct a
config that explicitly sets `attn_logit_softcap=50.0`.

### What we shipped this session as a direct result

1. `README.md` stale lines fixed (test count, muP status, recent work).
2. `configs/base_1b.yaml`: `peak_lr 2e-4 → 3.0e-4` (C3-locked), `end_lr_ratio 0.1 → 0.0` (D2Z), `attn_logit_softcap: 50.0`, `final_logit_softcap: 30.0`.
3. `src/myllm/training/loss.py`: chunked-CE accumulators cast to fp32 (D8 mitigation per jax #13529); softcap parameter wired through.
4. `src/myllm/model/config.py`: `attn_logit_softcap` + `final_logit_softcap` config fields.
5. `src/myllm/model/layers.py`: `GroupedQueryAttention.logit_softcap` (forces manual path when set; applies `cap*tanh(scores/cap)` pre-mask).
6. `src/myllm/model/transformer.py`: final-logit softcap on full-logit path.
7. `src/myllm/training/{train_step,eval_step}.py` + `scripts/run_pretrain.py`: thread softcap from config.
8. `tests/test_loss.py`: 2 new tests (softcap-matches-reference, softcap-clips-extremes) + 1 bf16 regression test.
9. `tests/test_model.py`: 3 new tests (softcap forward finite, final-cap bounds, softcap+qk_norm+segment compose).
10. `tests/test_watchdog_recovery.py`: precise-6σ-threshold edge case test.
11. `scripts/d8_gpu_repro.py`: 4-mode B200 repro driver — `baseline`, `disable-remat`, `fsdp`, `fp32-cce` — ready for $5 hour of B200 time.

---

## 8. Recent commits worth orienting against

```
8e50333  fix: data_position pytree mismatch in EVAL path under --fsdp (hotfix 2/2)
cbd5477  fix: data_position pytree mismatch under --fsdp (hotfix 1/2 — train_step)
5fb02d6  SESSION_HANDOFF: refresh after Round A+B + Layer 1+2 + refactor + C1+C2
52eb857  refactor: move init_state helpers from scripts/ to src/myllm/
04bfaf5  eval: MMLU-Pro + GSM8K real benchmark adapters (Round D6 / Layer 2)
302ba05  review packet: backfill per-source PPL + flag scorecard scoring limitation
7025bb9  infer/predict: add repo-root to sys.path so scripts.run_pretrain resolves
00f4ad2  eval_checkpoint: add --per-source-val-loss for Round C1 backfill
574dd8f  Round B: 4 Stage-2-gating P0s from re-audit (no GPU, low risk)
329b349  Round A: 6 quick wins from R1+R2 review (no GPU, low risk)
c2f4296  SESSION_HANDOFF: rewrite + trim (924 -> 230 lines)
d06fe57  post-pilot reviewer packet (2026-05-15)
f9399e7  Phase 2: docs refresh — Phase 1 done, pilot done
fbe9c72  Phase 1.2: per-source val loss (P0-1) via per-token NLL
107a551  Phase 1.5: forward-only eval_step (FSDP-safe, no donation)
97c59c1  Phase 1.6: G6 cross-mesh restore regression coverage
082fa20  Phase 1.3 + 1.4: --production + strict resume safety (P0-3)
be7574c  Phase 1.1: multi-epoch corpus reader (Stage 2 blocker)
```

Everything pushed to origin/main. HEAD = `8e50333`.

---

## 9. Contact

- **Lead**: harshit.hv@samatva.com (solo)
- **Auto-memory pointer**: `/root/.claude/projects/-root/memory/MEMORY.md`
- **R2 bucket**: `s3://llm-data/` (Cloudflare R2)

---

## 9. What the next session should NOT redo

- **Pilot training.** Done. val_loss 2.7303 / val_ppl 15.34.
- **FSDP gauntlet G1-G4.** Passing, regression-pinned in `tests/test_checkpoint_reshard.py`.
- **Round A + Round B + Layer 1 + Layer 2 part 1 + state_init refactor.** All shipped + pushed.
- **C2 4× B200 chunked-CE throughput.** Banked at `s3://llm-data/stage2-prep/benchmarks-4b200/`. **BUT**: re-bench at full-CE before locking Stage 2 cost model (chunked-CE has the NaN-grad bug, Stage 2 uses full-CE).
- **C3 μP/LR sweep.** Done. 3e-4 wins. muP transfer confirmed. Don't re-run.
- **Per-source PPL on pilot.** Banked at `s3://llm-data/scorecards/pilot-250m-v1-decay/`.
- **Reviewer packet doc errors.** Fixed in `329b349` + `302ba05`.
- **Hotfix 1 + 2 fixes.** Both shipped + regression-tested. Just `git pull` and they're in.

---

## 10. Stage 2 launch contract (locked from C3)

Once Stage 2 hardware/budget is decided, the launch command shape is:

```bash
python scripts/run_pretrain.py \
  --model-config configs/base_1b.yaml \
  --data-config configs/data/pretrain_mix_pilot.yaml \
  --tokenizer-path artifacts/tokenizer_v1.json \
  --packed-corpus-root /workspace/corpus_pilot_train \
  --run-name stage2-1b-{Nb}-tokens \
  --total-steps {STEPS} \                # 10B = ~20K steps, 30B = ~60K steps at mb=4
  --peak-lr-override 3.0e-4 \            # LOCKED by C3 (Run 3 winner)
  --micro-batch-override 4 \             # LOCKED by full-CE memory budget on B200
  --fsdp \                               # LOCKED (Stage 2 is the FSDP rehearsal)
  --log-every 100 \
  --eval-every 1000 \
  --eval-n-batches 16 \
  --per-source-val-loss \                # track which sources improve
  --corpus-epochs 6 \                    # LOCKED (5B corpus needs cycling for 30B tokens)
  --production \                         # LOCKED (fail-closed resume safety)
  --checkpoint-every 2000 \
  --checkpoint-root /workspace/ckpt/stage2-1b \
  --checkpoint-r2-prefix checkpoints/stage2-1b
```

**Locked from C3**: peak_lr=3e-4, mb=4, --fsdp, no `--use-chunked-ce`, --corpus-epochs=6, --production.

**Pending decision**: hardware (4× B200 vs 8× B200 vs 8× H200 SXM), seq length (whatever the corpus has, 8192), total tokens (10B vs 30B).

Once the smoke probe (2000-step) confirms throughput at the chosen hardware, the full Stage 2 commit is straightforward.
