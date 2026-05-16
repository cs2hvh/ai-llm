# Session Handoff (current — last refreshed 2026-05-16, evening)

> Filename keeps the `2026-05-14` suffix for auto-memory continuity, but
> this doc is **the live handoff for the next session**. Older
> minute-by-minute pilot narrative lives in `pilots/250m_v1/TIMELINE.md`;
> verified architecture/state in `docs/PROJECT_OVERVIEW.md`; pilot
> artifacts inventory in `pilots/250m_v1/R2_PATHS.md`.

---

## 0. TL;DR

1. **Stage 1 pilot DONE** (250M, $385, val_loss 2.7303 / val_ppl 15.34). DP-replicated on 4×H200, NOT FSDP. Final ckpt on R2.
2. **Phase 1 engineering DONE** (5 commits, +32 tests). All pre-pilot reviewer P0/P1s closed.
3. **2 reviewer rounds processed + closed code-side**: Round A (6 quick wins), Round B (4 P0s incl. KD vocab gate, watchdog data_position fix, benchmark FSDP path, scorecard predict_fn). Suite 642 → 737.
4. **Layer 1 done** — reviewer packet §2.4 backfilled with real per-source PPL from the C1 GPU run; §2.6 documents the still-placeholder scorecard scoring.
5. **Layer 2 part 1 done** — MMLU-Pro + GSM8K real benchmark adapters with proper scoring + 37 tests. IFEval / HumanEval+ / MBPP+ remain on the placeholder path (separate-PR tracked).
6. **Refactor done** — `init_model_and_optimizer` / `initial_train_state` / `ensure_tokenizer_local` / `resolve_wsd_schedule_params` moved from `scripts/` to `src/myllm/`. Removed sys.path hack.
7. **C1 partial** — per-source PPL banked on R2 (`s3://llm-data/scorecards/pilot-250m-v1-decay/`). Scorecard benchmark run aborted (placeholder scoring → 1.0 accuracy). ~$8 spent.
8. **C2 DONE** — 4× B200 NVLink-5 (Vast.ai). **30% MFU at seq=8192 / 46% MFU at seq=4096**. Both above Stage-2 bars. Stage 2 wall projection at 30B tokens: ~25-38 hr, ~$350-530.
9. **C3 in flight, transitioning Vast.ai → RunPod** because user hit issues on Vast.ai. C2 results uploaded to R2 first. C3 commands ready, hardware-agnostic.
10. **DO NOT** commit Stage 2 ($350-700) before C3 decision gate. The μP/LR sweep is the last gating piece.

---

## 1. Where things stand (2026-05-16, evening)

### Pilot (unchanged)

| Item | Value |
|---|---|
| Sharding | **DP-replicated** on 4×H200 SXM (no `--fsdp` in launch). FSDP proven separately via gauntlet G1-G4 + L2/L3 canaries. |
| Stage 1 final | step 151,990, val_loss 2.8776, val_ppl 17.77, corpus exhausted single-pass |
| Stage 1.5 final | step 171,990, val_loss 2.7303, val_ppl 15.34. Δ = -0.147 nats from decay |
| Watchdog | Quiet — 288 NaN-skips / 172K steps, 0 hard rollbacks |
| W&B runs | `roydqofb` (pre-crash) + `u5xsxm0l` (post-resume) + `pxoungh9` (Stage 1.5 decay) |
| Per-source PPL | **MEASURED 2026-05-16** — banked at `s3://llm-data/scorecards/pilot-250m-v1-decay/pilot-250m-v1-decay-per-source.json`. Aggregate 2.734 / 15.40. Code = best (PPL 3.30), Hindi = worst (PPL 41.32). Full table in `docs/review/POST_PILOT_REVIEW_2026-05-15.md` §2.4. |

### Phase 1 engineering (all DONE — from previous handoff)

| # | Commit | What |
|---|---|---|
| 1.1 | `be7574c` | Multi-epoch corpus reader |
| 1.3+1.4 | `082fa20` | `--production` flag + strict packed-resume safety |
| 1.5 | `107a551` | Forward-only `make_eval_step` (FSDP-safe) |
| 1.6 | `97c59c1` | G6 cross-mesh restore regression tests |
| 1.2 | `fbe9c72` | Per-source val loss via per-token NLL |

### Reviewer cycle (Rounds A + B + Layer 1 + Layer 2 part 1 + refactor — all SHIPPED)

| Commit | What | Source |
|---|---|---|
| `329b349` | **Round A** 6 quick wins: 3 packet doc fixes, eval_checkpoint jax-order, governance template fix, canary_ladder packed-L3 default, version pins + orbax compat smoke test (10 tests) | R1+R2 P1s |
| `574dd8f` | **Round B** 4 Stage-2-gating P0s: KD vocab fail-closed gate, watchdog `_recover_from_spike` template+data_position fix, `benchmark_throughput.py --fsdp`, scorecard `predict_fn` via shared `src/myllm/infer/predict.py` (16 tests) | R1+R2 P0s |
| `7025bb9` | infer/predict sys.path fix (hit on the C1 pod) | runtime |
| `00f4ad2` | `eval_checkpoint.py --per-source-val-loss` for the C1 backfill | C1 prep |
| `302ba05` | **Layer 1** packet §2.4 backfill (real per-source numbers) + §2.6 scorecard placeholder note | post-C1 |
| `04bfaf5` | **Layer 2 part 1** MMLU-Pro + GSM8K real benchmark adapters with proper scoring (37 tests) | post-C1 |
| `52eb857` | **Refactor** init_state helpers → `src/myllm/training/state_init.py`; `ensure_tokenizer_local` → `src/myllm/utils/storage.py`; removed sys.path hack | hygiene |

### C1 (1× H200 / H100) — DONE PARTIAL

- **Got**: per-source val_loss for all 13 sources, banked to R2.
- **Abandoned**: release scorecard run — placeholder scoring policy ("non-empty output = success") gives accuracy 1.0 on every sample. Round-1 reviewer flagged this; Round B4 wired the predict_fn but the per-benchmark `score()` methods are still scaffolds for IFEval/HumanEval+/MBPP+. Layer 2 part 1 fixed MMLU-Pro + GSM8K but didn't ship in time for the C1 run.
- **Spent**: ~$8 (~30 min waste on the doomed scorecard before aborting).

### C2 (4× B200 NVLink-5 on Vast.ai) — DONE

Real-world FSDP throughput on the path Stage 2 will actually run. **Both results uploaded to R2 at `s3://llm-data/stage2-prep/benchmarks-4b200/`** (user uploaded before tearing down Vast.ai).

| Combo | Step time | Aggregate tok/s | Real BF16 MFU | Peak HBM | Stage 2 (30B) wall | Stage 2 (30B) $ |
|---|---|---|---|---|---|---|
| seq=8192, mb=16 | 2.37 s | 221,246 | **~30%** | 131 GB / 183 GB | 37.7 hr | ~$530 |
| seq=4096, mb=16 | 0.78 s | 335,683 | **~46%** | 43.7 GB / 183 GB | 24.8 hr | ~$350 |

Notes:
- The benchmark's printed `MFU estimate: 8.2% / 12.5%` is **cosmetic** — defaults `peak_flops_bf16` to an FP8-or-sparse number (~1979 TFLOPS). Real BF16-dense MFU vs B200's 1100 TFLOPS peak is **~30% / ~46%**, both above Stage-2 thresholds.
- 4× B200 NVLink-5 (NV18 mesh) ≈ 1.8 TB/s peer-to-peer — actually BETTER per-link than H200's NVLink-4.
- Synthetic-data NaN loss starting at step 25 on the 8K run is the **expected** "no LR warmup + random tokens" pattern, doesn't invalidate the throughput measurement. Confirmed by the 4K run NOT NaN'ing (smaller per-step batch tokens → smaller grad magnitudes).
- **Decision implication**: 4K is materially better economics for Stage 2 (1.5× faster, $180 cheaper at 30B). Trade-off is recipe consistency with pilot (8K). The Stage 2 *rehearsal* can tolerate either; the Stage 3 base run should decide based on quality not just throughput.

### C3 (μP/LR sweep) — IN FLIGHT, moving Vast.ai → RunPod

Vast.ai instance had issues — user moving to RunPod. C3 is hardware-agnostic (LR transfer is a model property). Commands prepared:

- 3 sequential runs of `scripts/run_pretrain.py` at peak_LR × {0.5, 1.0, 1.5} of `configs/base_1b.yaml`'s `peak_lr: 2.0e-4`.
- LR values tested: **1.0e-4, 2.0e-4, 3.0e-4** (3e-4 matches the pilot's peak_lr — informative whether muP transfer holds from 250M to 1B).
- 1000 steps each. Wall on 4× B200 was projected ~42 min each = ~2 hr total + JIT. On 8× H200 SXM would be slightly faster.
- Flags: `--fsdp --use-chunked-ce --corpus-epochs 6 --production --per-source-val-loss --eval-every 200 --no-wandb`.
- Compare the val_loss at step 1000 across the 3 runs. Best LR wins. If 3e-4 NaN-spirals, that's the *signal* the LR ceiling is below the pilot's value.

---

## 2. The plan ahead

### Immediately (the user's current RunPod session)

1. **Bring up RunPod pod** — paste `nvidia-smi --query-gpu=...` + `nvidia-smi topo -m` + python version, get exact commands.
2. **Bootstrap** — apt deps, clone, venv with `python3.11` or `python3.12`, `pip install -e ".[cuda]"`, verify GPUs visible to JAX with a matmul probe.
3. **Pull artifacts** from R2 (tokenizer + composed pilot corpus, ~20 GB).
4. **Phase D (C3)** — 3-LR sweep, ~2 hr wall, ~$28-50 GPU cost.
5. **Upload C3 logs + final checkpoints** to `s3://llm-data/stage2-prep/mup-sweep-{hw}/`.

### Round C5 — Stage 2 rehearsal (after C3 numbers are in)

Decision gate: pick the LR with the best val_loss at step 1000 + decide seq=4096 vs 8192 from C2 cost-benefit + commit.

- 1B at **10-30B tokens** on whichever 4-or-8 GPU pod is most economical.
- Cost: **$350-700** on 4× B200, **$700-2K** on 8× H200 SXM (much faster wall, same total cost).
- Flags: `--corpus-epochs 6 --production --fsdp --use-chunked-ce --per-source-val-loss --checkpoint-every 2000 --eval-every 1000`.
- This is the *real* rehearsal — produces a 1B checkpoint that's the input to Phase 5 (Stage 3 prep + launch).

### Layer 2 parts 2 + 3 — remaining benchmark adapters (CPU, deferred)

| Item | Effort | Status |
|---|---|---|
| IFEval adapter | ~3-4 hr | Deferred — needs internet to verify `google/IFEval` 25-constraint schema accurately. Wrong constraint logic is worse than no scorer. |
| HumanEval+ / MBPP+ adapters | ~3-4 hr each | Deferred — need sandboxed code-exec subprocess + test runner. Security-sensitive. |

Once both land, **a $30 H100 PCIe pod re-runs the scorecard against the pilot checkpoint** and we have real benchmark numbers for the model card.

### Round D — Stage 3 prep (parallel with Stage 2 rehearsal)

| # | Item | Effort | Status |
|---|---|---|---|
| D1 | Chunked distillation (student CE+logZ through chunked path; teacher KL on top-K only) | 1-2 days | Pending |
| D2 | Teacher top-K mass audit on real text | 0.5 day + GPU | Pending |
| D3 | Stratified per-source held-out (sample K per source across shards, not corpus head) | 4 hr | Pending |
| D4 | pg19 replacement for Stage 3 corpus | depends | Pending |
| D5 | Stack Exchange `question + chosen_response` schema fix | 2 hr | Pending |
| D6 (in progress) | Real scoring policies — MMLU-Pro+GSM8K DONE, IFEval/HumanEval+/MBPP+ pending | 1-2 days remaining | Partial |
| D7 | Logical-axis FSDP sharding rules in `mesh.py` | 2-3 days | Pending |

---

## 3. Critical gotchas (silent-corruption modes to remember)

Same as previous handoff — patterns to carry forward:

1. **int32 cursors in JIT'd state** — Python ints default to int32 under JAX tracing; wrapped at ~step 65,500 in pilot. Fix: pop from state before JIT call.
2. **Single-pass iterators that look infinite** — `iter_packed_pairs` stopped at `total_sequences`. Stage 2 MUST launch with `--corpus-epochs 6+`.
3. **Orbax API drift bites in minor versions** — pin exactly (jax==0.4.38, orbax==0.7.0, tensorstore==0.1.83 in pyproject.toml). Smoke-test restore kwargs after any version bump (`tests/test_orbax_api_compat.py`).
4. **FSDP `donate_argnums=(0,)` invalidates state for any subsequent caller** — eval-during-training under FSDP needs `make_eval_step` (forward-only, no donation).
5. **Pilot ran DP-replicated, not FSDP.** Do not let prose drift on this. FSDP proven by gauntlet G1-G4 separately. Round A1 (commit `329b349`) corrected this in the reviewer packet.
6. **Scorecard scoring is still placeholder for IFEval/HumanEval+/MBPP+** — Layer 2 only fixed MMLU-Pro + GSM8K. Treat scorecard.json numbers for those benchmarks as "model produces output", not "model produces correct output".
7. **JAX x32 mode demotes int64 in `restore_args` path** — known limitation, pinned by `tests/test_orbax_api_compat.py::test_per_leaf_restore_args_demotes_int64_above_2_31`. Production training-resume uses the legacy restore path which preserves int64; the per-leaf path is only used for cross-mesh inference where data_position is irrelevant.

---

## 4. Reviewer findings status

**All P0/P1 from R1+R2 are closed code-side**:

| Finding | Status | Commit |
|---|---|---|
| KD vocab mismatch (P0) | CLOSED | `574dd8f` B1 |
| Watchdog `_recover_from_spike` template + data_position (P0) | CLOSED | `574dd8f` B2 |
| `benchmark_throughput.py` doesn't measure FSDP (P0) | CLOSED | `574dd8f` B3 |
| Scorecard `predict_fn` NotImplementedError (P0) | CLOSED | `574dd8f` B4 (predict_fn) + `04bfaf5` (MMLU-Pro + GSM8K scoring) |
| `eval_checkpoint.py` jax import order (P1) | CLOSED | `329b349` A3 |
| `render_governance_cards.py` template names (P1) | CLOSED | `329b349` A4 |
| `canary_ladder.py` packed-L3 default (P1) | CLOSED | `329b349` A5 |
| Doc errors in reviewer packet (P1) | CLOSED | `329b349` A1 + `302ba05` Layer 1 |
| Orbax/JAX/TensorStore exact pins (P1) | CLOSED | `329b349` A6 |
| Per-source PPL still TBD in packet (R2) | CLOSED | `302ba05` Layer 1 |

**Still in deferred-with-rationale state**:

| Finding | Status |
|---|---|
| HumanEval+ / MBPP+ scoring policy | Pending Round D6 (sandboxed code exec) |
| IFEval scoring policy | Pending Round D6 (programmatic constraint checks) |
| Chunked distillation memory savings in decay phase | Pending Round D1 |
| Stratified per-source held-out (held-out is corpus head, not shard-stratified) | Pending Round D3 |
| Logical-axis FSDP sharding rules | Pending Round D7 |

---

## 5. Resume / verify

```bash
cd /root/llm-build
.venv/bin/python -m pytest -q
# expect: 737 passed, 1 skipped
```

For pilot checkpoint pull + post-hoc per-source eval (the C1 path we ran on 1× H200):

```bash
# tokenizer
aws --endpoint-url "$S3_ENDPOINT_URL" s3 cp \
  s3://$S3_BUCKET/tokenizer/myllm-spm-unigram-131k-v2.json \
  artifacts/tokenizer_v1.json

# pilot final checkpoint
mkdir -p /workspace/ckpt/pilot-250m-v1-decay
aws --endpoint-url "$S3_ENDPOINT_URL" s3 sync \
  s3://$S3_BUCKET/checkpoints/pilot-250m-v1-decay/step-000171990/ \
  /workspace/ckpt/pilot-250m-v1-decay/step-000171990/

# composed pilot corpus (~20 GB)
mkdir -p /workspace/corpus_pilot_train
aws --endpoint-url "$S3_ENDPOINT_URL" s3 sync \
  s3://$S3_BUCKET/corpus_v1_pilot/train/ /workspace/corpus_pilot_train/

# per-source eval
KERAS_BACKEND=jax .venv/bin/python scripts/eval_checkpoint.py \
  --checkpoint /workspace/ckpt/pilot-250m-v1-decay/step-000171990 \
  --model-config configs/pilot_250m.yaml \
  --tokenizer-path artifacts/tokenizer_v1.json \
  --packed-corpus-root /workspace/corpus_pilot_train \
  --micro-batch 4 --n-batches 64 --per-source-val-loss \
  --output-json artifacts/scorecards/pilot-rerun-per-source.json
```

For C2 + C3 re-run on a fresh pod, see `pilots/250m_v1/COMMANDS.md` for the pilot launch template + the per-LR runs follow the same shape with `--peak-lr-override`.

---

## 6. R2 paths (Stage 2 prep artifacts so far)

| What | R2 path |
|---|---|
| Per-source PPL JSON (C1) | `s3://llm-data/scorecards/pilot-250m-v1-decay/pilot-250m-v1-decay-per-source.json` |
| 4× B200 throughput (C2) | `s3://llm-data/stage2-prep/benchmarks-4b200/1b_4b200_seq{4096,8192}_mb16.{json,log}` |
| μP/LR sweep (C3) — TBD | `s3://llm-data/stage2-prep/mup-sweep-{hw}/lr{0_5,1_0,1_5}x.log` |
| μP/LR sweep checkpoints — TBD | `s3://llm-data/stage2-prep/checkpoints/mup-sweep-{hw}/` |
| Pilot artifacts (unchanged) | See `pilots/250m_v1/R2_PATHS.md` |

---

## 7. Where things live

| Need | Open |
|---|---|
| Pilot results, R2 paths, frozen configs, COMMANDS | `pilots/250m_v1/` |
| Canonical project state | `docs/PROJECT_OVERVIEW.md` |
| Reviewer packets + responses | `docs/review/POST_PILOT_REVIEW_2026-05-15.md` (kept current — §2.4 has real numbers, §2.6 documents scorecard limitation) |
| Stage 3 Rust migration plan | `docs/stage3_rust_migration_plan.md` |
| muP design + scaling rules | `docs/mup_design.md` |
| Teacher distillation strategy | `docs/teacher_distillation_strategy.md` |
| Pilot config (250M, 768/3072, GQA 3:1) | `configs/pilot_250m.yaml` |
| Base 1B config (2048/8192, GQA 4:1, peak_lr 2e-4) | `configs/base_1b.yaml` |
| Stage 1.5 decay config | `configs/pilot_250m_decay.yaml` |
| Per-source val loss (in-training) | `src/myllm/training/eval_hook.py` + `eval_step.py` |
| Per-source val loss (post-hoc CLI) | `scripts/eval_checkpoint.py --per-source-val-loss` |
| G6 cross-mesh restore | `src/myllm/training/checkpoint.py` |
| FSDP sharding | `src/myllm/training/mesh.py` (shape-heuristic; Round D7 will replace with logical-axis) |
| Watchdog + recovery | `src/myllm/training/watchdog.py` + `loop.py:_recover_from_spike` |
| Init state helpers (refactored to src/) | `src/myllm/training/state_init.py` |
| Tokenizer fetch | `src/myllm/utils/storage.py::ensure_tokenizer_local` |
| Shared inference path (predict_fn for scorecard / generate / eval) | `src/myllm/infer/predict.py` |
| MMLU-Pro adapter (proper scoring) | `src/myllm/eval/benchmarks/mmlu_pro.py` |
| GSM8K adapter (proper scoring) | `src/myllm/eval/benchmarks/gsm8k.py` |
| Orbax API compat smoke test | `tests/test_orbax_api_compat.py` |

---

## 8. Recent commits worth orienting against

```
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

Everything pushed to origin/main. `737 passed, 1 skipped` is the suite baseline.

---

## 9. Contact

- **Lead**: harshit.hv@samatva.com (solo)
- **Auto-memory pointer**: `/root/.claude/projects/-root/memory/MEMORY.md`
- **R2 bucket**: `s3://llm-data/` (Cloudflare R2)

---

## 10. What the next session should NOT redo

- **Pilot training.** Done. val_loss 2.7303 / val_ppl 15.34.
- **FSDP gauntlet G1-G4.** Passing on 2× H200 SXM, regression-pinned in `tests/test_checkpoint_reshard.py`.
- **Round A + Round B + Layer 1 + Layer 2 part 1 + state_init refactor.** All shipped + pushed.
- **C2 4× B200 throughput.** Banked at `s3://llm-data/stage2-prep/benchmarks-4b200/`. Don't re-bench on the same hardware.
- **Per-source PPL on the pilot checkpoint.** Banked at `s3://llm-data/scorecards/pilot-250m-v1-decay/pilot-250m-v1-decay-per-source.json`.
- **The reviewer packet doc errors (FSDP claim, ffn_dim, base 1B shape).** Fixed in `329b349` + `302ba05`.
- **Reviewer's 12-question §8 from `POST_PILOT_REVIEW_2026-05-15.md`.** R2 audit answered most; Layer 1+2 closed the doc-side concerns.
