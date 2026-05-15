# MyLLM Pilot 250M v1 — Timeline

Day-by-day chronological narrative of how the pilot got built. Reference commit hashes are anchored to the main repo at `github.com/cs2hvh/ai-llm`.

## 2026-05-10 — Project bootstrap

Architecture review against Llama 3.2 1B, SmolLM2 1.7B, Qwen 2.5 1.5B. Locked decisions:
- Dense Llama-style decoder-only, 16 layers × hidden 768 → ~250 M params
- Keras 3 + JAX backend (deterministic compute, native multi-device sharding)
- WSD (Warmup-Stable-Decay) LR schedule per SmolLM2 / MiniCPM playbook
- muP HP-transfer (base_width 256) so wind-tunnel HPs port to base 1 B
- Cloudflare R2 as the durable artifact store
- $15M total project ceiling

Wrote `PLAN.md` (~600 lines), set up repo scaffold at `/root/llm-build`. Phase 0 synthetic-data smoke ran on CPU.

## 2026-05-11 — Tokenizer + recipe upgrades

HF Unigram tokenizer hit ~15 GB scale ceiling (silent OOM); swapped to native SentencePiece via `scripts/train_tokenizer_spm.py`. Final artifact `myllm-spm-unigram-131k-v2.json` (4.79 MB, 131,072 vocab) → R2.

Recipe upgrades:
- WSM (Warmup-Stable-Merge) checkpoint averaging spec
- muP scaled_init_for_residuals on (matches base for HP transfer)
- QK-norm in attention (consistent with base + wind_tunnel)
- z-loss coef=1e-4

First H200 pod brought up; 10-cell wind-tunnel LR sweep launched. NaN losses appeared at step ~4100 — root-caused to GQA + DecoderBlock mask propagation. Fixed by routing through the attention layer's mask correctly.

## 2026-05-12 — Reviewer audit + teacher re-verification

Reviewer Round 3 (the friend) flagged:
- License clauses on Mistral-Medium-3.5 (forbids competitor training)
- Qwen 3.6 added a chat-only modality restriction

WebFetch verification confirmed both. **Teacher plan v2 locked**:
- DeepSeek-V4-Pro-Base + Olmo-3-32B-Base (both base, no chat-only restrictions)
- Top-K = 64 logits per teacher per token (offline cached)
- Decay-phase distillation only (last 15%) — ~22 TB cache on R2

Throughput planning baseline locked at **280-360K tok/sec aggregate on 8×H200** (not the 520K stretch).

Token budget locked at 600 B for Stage 3 base run (down from initial 1 T, after compute audit).

## 2026-05-13 (early) — FSDP validation + corpus build kickoff

FSDP gauntlet G1-G4 validated on 2×H200 SXM:
- G1 boot + 3 steps ✓ PASS
- G2 HLO inspection: `reduce_scatter=46, all_reduce=22` ✓ PASS (FSDP genuinely sharding, not silent DDP)
- G3 L2 parity canary `|Δloss| ≤ 5e-3` over 50 steps ✓ PASS (DP vs FSDP numerically equivalent)
- G4 memory savings: FSDP=1617 MB vs DP=14803 MB = 89% savings ✓ PASS

G5 (throughput) deferred to Stage 1 measurement; G6 (cross-mesh reshard) known broken — deferred to Stage 2 prep.

Pilot corpus rebuild plan written (`docs/pilot_corpus_rebuild_plan.md`): 5 B-token corpus at seq_len=8193 so the same data serves pilot (ctx=8192) + Stage 2/3 base v1. Codeparrot substituted for gated starcoderdata.

`scripts/run_parallel_builds.py --target-tokens-per-source 5_000_000_000 --max-parallel 8` launched at 17:39 UTC on the 128-core dev box.

## 2026-05-13 (evening) — Corpus build completes

Wall time: 2h 1m. 13 sources @ ~5.0 B tokens total, all uploaded to `s3://llm-data/corpus_v1_pilot/sources/`. fineweb_edu finished at 19:50:39 UTC.

**Compose pass** (`scripts/compose_mixed_corpus.py --strict-sources`) ran for 1h 49m. Output: 608,088 sequences (4.98 B tokens) across 10 shards, 19 GB. Max source-share drift **0.33%** (under the 2% L5 threshold — passes). Uploaded to `s3://llm-data/corpus_v1_pilot/train/`.

Pilot dry-run on synthetic data (CPU): 5 steps, eval at step 2 + 4, checkpoint saved → cleanly. Watchdog stress test (5/5 unit tests) PASS.

## 2026-05-14 (00:03 UTC) — STAGE 1 PILOT LAUNCHED

4×H200 SXM pod from RunPod. Hit NCCL initialization bug — fixed with `NCCL_NVLS_ENABLE=0` + `NCCL_IB_DISABLE=1` exports. Smoke test on real corpus passed. Pilot launched at **2026-05-14T00:03:38Z** via:

```bash
python scripts/run_pretrain.py \
    --model-config configs/pilot_250m.yaml \
    --data-config configs/data/pretrain_mix_pilot.yaml \
    --tokenizer-path artifacts/tokenizer_v1.json \
    --packed-corpus-root /workspace/corpus_pilot_train \
    --run-name pilot-250m-v1-2026-05-13 \
    --total-steps 229000 \
    --micro-batch-override 4 \
    --log-every 100 --eval-every 5000 --eval-n-batches 32 \
    --checkpoint-every 5000 \
    --checkpoint-root /workspace/ckpt/pilot-250m-v1 \
    --checkpoint-r2-prefix checkpoints/pilot-250m-v1
```

W&B: `pilot-250m-v1-2026-05-13` run id `roydqofb`. Initial loss ~11.81 (matches `log(131072) = 11.78` random-init expectation).

## 2026-05-14 (07:06 UTC) — int32 overflow crash

Crashed at step **~65,500** with `OverflowError: Python int 2147483648 too large to convert to int32`. Root cause: `state["data_position"]` exceeds `2^31 = 2,147,483,648` (at mb=4 × seq=8192 = 32,768 tokens/step, exactly at step 65,536). JAX default-types Python ints as int32 when tracing JIT'd functions.

Loss at crash: ~2.87 smoothed (down from 11.81 in 7 hours — healthy descent). 142 NaN-skip events handled cleanly during the 65K steps; no `hard_spike` rollbacks.

## 2026-05-14 (~11:00 UTC) — Resume after fix

**Commit `9f442f7`**: pop `data_position` from a shallow state copy before each `train_step_fn` call; restore as Python int after. data_position isn't used INSIDE train_step (just carried through state for checkpointing). Same fix needed for `eval_hook` (caught later — commit `dd7b202`).

Pilot resumed from step-65,000 checkpoint cleanly. W&B opened a SECOND run `u5xsxm0l` (the resume didn't continue the original `roydqofb`). Both finalized cleanly; curves split between them.

## 2026-05-14 (20:17 UTC) — STAGE 1 COMPLETED at step 151,990

Corpus exhausted before reaching `--total-steps 229000`. Math: 608,088 corpus sequences / 4 per step = 152,022 steps to exhaust. Stopped at 151,990 (32 short due to the eval-hook held-out batches reserved at startup).

WSD decay phase **NEVER reached** (was scheduled for step 194,650 = 85% of 229K). Model trained at full peak LR (3e-4) throughout the stable phase, then stopped.

`training_complete` event fired cleanly at 2026-05-14T20:17:34Z. Final checkpoint saved + mirrored to R2 (~2.65 GB, 18 files).

## 2026-05-14 (21:50 UTC) — Post-hoc eval Stage 1

Built `scripts/eval_checkpoint.py` (commit `70b9009`) to compute val_loss/val_ppl on any saved checkpoint without re-launching training. Ran on the same pod:

```
val_loss : 2.877632
val_ppl  : 17.7721
n_batches: 32
checkpoint: step-000151990
```

Borderline — pilot is "working but under-converged" because decay never ran.

## 2026-05-14 (22:33 UTC) — STAGE 1.5 LAUNCHED

**Commit `dd7b202`** added:
- `LoopConfig.reset_data_position_on_resume: bool`
- `--reset-data-position-on-resume` CLI flag in `run_pretrain.py`
- int32 fix in `eval_hook.py` (eval was silently failing post-resume in Stage 1)
- `configs/pilot_250m_decay.yaml`: warmup_steps=0, decay_fraction=0.1163 so total_steps=171,990 puts decay onset at step 151,990

Pre-staged step-151,990 checkpoint into a fresh dir (`/workspace/ckpt/pilot-250m-v1-decay/`) and launched:

```bash
python scripts/run_pretrain.py \
    --model-config configs/pilot_250m_decay.yaml \
    --data-config configs/data/pretrain_mix_pilot.yaml \
    --tokenizer-path artifacts/tokenizer_v1.json \
    --packed-corpus-root /workspace/corpus_pilot_train \
    --run-name pilot-250m-v1-decay-2026-05-14 \
    --total-steps 171990 \
    --micro-batch-override 4 \
    --log-every 100 --eval-every 1000 --eval-n-batches 32 \
    --checkpoint-every 2000 \
    --checkpoint-root /workspace/ckpt/pilot-250m-v1-decay \
    --checkpoint-r2-prefix checkpoints/pilot-250m-v1-decay \
    --reset-data-position-on-resume
```

W&B run id `pxoungh9`. 20K new steps at decaying LR (3e-4 → 3e-5 linear). Eval hook fired ~20 times (~every 1000 steps) — the int32 fix was confirmed working in production.

## 2026-05-15 (00:51 UTC) — STAGE 1.5 COMPLETED at step 171,990

Wall: 2h 18m. `training_complete` at 2026-05-15T00:51:05Z. No NaN events, no spikes, clean run. Final checkpoint manifest `"reason": "final"` (verified on R2).

Post-decay eval (same eval_checkpoint.py + same held-out batches):

```
val_loss : 2.730350         ← -0.147 nat improvement
val_ppl  : 15.3383          ← -13.7% perplexity reduction
n_batches: 32
checkpoint: step-000171990
```

**This is the official pilot final number for the model card.**

## 2026-05-15 (~19:40 UTC) — Generation testing on 1×H100

User tore down the 4×H200 pod; spun up a 1×H100 for inference smoke testing. Hit the G6 reshard bug immediately: checkpoint saved on 4 devices, restoring on 1 device → `RestoreArgs(sharding=...)` was missing.

Three commits to fix:
- `13d6126`: G6 reshard fix v1 (had stale orbax 0.6 `shape` kwarg)
- `3be12de`: drop unsupported `shape` kwarg (orbax 0.7 API drift)
- `ca1c40b`: force sharding on ALL leaves (orbax saves Python scalars as 0-d arrays too)

Generation ran on 1×H100 with `scripts/generate.py` (new file, commit `bc1d2b1`). 10 default prompts ran in ~3 minutes (~$1 of compute). Results documented in `RESULTS.md` § "Sample generations."

**Key findings**:
- Domain awareness works (Wikipedia → encyclopedia, code → Python, math → LaTeX, Hindi → Hindi)
- Multilingual confirmed (sangraha → coherent Hindi)
- Factual recall weak ("capital of France" doesn't reliably produce "Paris")
- Math fails (250M can't do arithmetic)
- Loops after ~30-50 tokens (expected for small base model without repetition penalty / SFT)

Pilot generation verified working. Recipe validated.

## 2026-05-15 (~20:00 UTC) — All pods torn down

4×H200 pod terminated. 1×H100 pod terminated. All artifacts on R2 (161.30 GB verified). $0/hr compute from here.

`docs/SESSION_HANDOFF_2026-05-14.md` updated with §14 (post-pilot final state). Auto-memory pointer refreshed.

## 2026-05-15 (~21:00 UTC) — Pilot folder created

This folder (`pilots/250m_v1/`) created as the archival package for Stage 1. Self-contained: README + RESULTS + TIMELINE + COMMANDS + R2_PATHS + configs + small artifacts.

**Stage 1 closed.** Next: Phase 1 engineering (multi-epoch corpus reader + reviewer P0s), CPU-only work. See `docs/SESSION_HANDOFF_2026-05-14.md` §14.9 for the full roadmap.

## Key commits in chronological order

| Date | Commit | Subject |
|---|---|---|
| 2026-05-10 | (early) | Project scaffold, PLAN.md |
| 2026-05-11 | (multi) | Tokenizer SPM swap, recipe upgrades |
| 2026-05-12 | `9c5d494` | Teacher plan v2: drop Mistral + Qwen3.6, lock Olmo-3-32B |
| 2026-05-13 | `cc56daa` | Decontam DualMode (8+13 gram) wired |
| 2026-05-13 | `03a8b3a` | Pilot corpus rebuild plan + pretrain_mix_pilot.yaml |
| 2026-05-13 | `6cf299e` | Canonical single-venv recipe: torch 2.7.1 + jax[cuda12] 0.4.38 |
| 2026-05-13 | `7331377` | D2 decision: bump pilot ctx 4096 → 8192 |
| 2026-05-13 | `518aa50` | eval-during-training: --eval-every wires val_loss + ppl |
| 2026-05-13 | `fb6a537` | release scorecard scaffold (predict_fn STUB) |
| 2026-05-14 | `9f442f7` | int32 overflow fix for data_position in train_step |
| 2026-05-14 | `70b9009` | scripts/eval_checkpoint.py — post-hoc eval |
| 2026-05-14 | `dd7b202` | Stage 1.5 scaffolding + eval_hook int32 fix |
| 2026-05-14 | `d3ac216` | SESSION_HANDOFF §13 post-pilot update |
| 2026-05-15 | `bc1d2b1` | scripts/generate.py — autoregressive generation |
| 2026-05-15 | `13d6126`, `3be12de`, `ca1c40b` | G6 reshard fix (3-commit story) |
| 2026-05-15 | `d4b9b50` | SESSION_HANDOFF §14 — post-pilot + post-Stage-1.5 + post-G6 |
| 2026-05-15 | (this) | pilots/250m_v1/ folder creation |

## Lessons that shape Stage 2 + 3 plans

From this timeline, things that proved important:

1. **Always pop large-integer accumulators from JAX-tracked state** — data_position int32 overflow could have been caught earlier with a unit test. **Add for Stage 2: tests/test_long_run_simulation.py that mocks 100K+ steps.**

2. **The pilot corpus is single-epoch only at our batch size** — Stage 2's 30B-token target needs **6× more corpus OR multi-epoch reader**. Phase 1.1 work.

3. **In-training eval hook silently fails on bugs** — eval failures should be LOUDER. The int32 bug in `eval_hook.py` went unnoticed for ~85K steps because the except clause was too forgiving.

4. **Cross-mesh restore (G6) is required for ANY post-training operation** — fixed now, but should have been done before Stage 1 inference. Phase 1 should add `tests/test_checkpoint_reshard.py` to prevent regression.

5. **WSD decay produces ~0.15 nats improvement, not 0.4** — set realistic expectations for Stage 2/3.

6. **Generation testing should be part of the pilot scorecard** — sample outputs caught the model's actual behavior in ways perplexity alone didn't (looping, factual weakness, multilingual success). Add this to Stage 2 acceptance.

7. **The Stage 1.5 pattern (decay-only continuation pass) is useful** — even if Stage 2/3 have decay built into the schedule, the "load checkpoint + run decay-only" pattern via `--reset-data-position-on-resume` is now a tool in the toolbox.
