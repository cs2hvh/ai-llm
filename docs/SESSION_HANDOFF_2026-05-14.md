# Session Handoff — 2026-05-14

**For the next session (you or another Claude) picking this up.** Read this first; it's the canonical resume pointer for whatever state the project is in.

**Last update:** 2026-05-15 ~20:00 UTC, **post-pilot, post-Stage-1.5, post-generation-test**. Pod has been torn down (or is being torn down) — all artifacts safely on R2.
**Author:** Session that ran the pilot end-to-end + Stage 1.5 + first generation test. Replaces `docs/SESSION_STATE_2026-05-13.md`. Sections §0–§12 are the original handoff (still valid for orientation); §13 covers post-pilot; **§14 covers Stage 1.5 + generation verification + G6 fix + tear-down**.

---

## 0. TL;DR — read this if nothing else (UPDATED 22:00 UTC)

- **Project**: MyLLM — solo-lead enterprise effort to train a 1B-param decoder-only LLM from scratch. Llama-style, JAX/Keras 3, 131k vocab, ctx=8192.
- **Pilot is DONE.** Stage 1 (250M model) trained to step **151,990 of planned 229,000** — corpus exhausted before reaching the planned step count.
- **Final loss**: train smoothed ~2.4, **val_loss 2.878, val_ppl 17.77** (post-hoc eval via `scripts/eval_checkpoint.py` at commit `70b9009`).
- **WSD decay phase never reached** — corpus exhausted in the stable phase. Model trained at peak LR throughout, never got the "settle into local minimum" benefit.
- **Stage 1.5 decay-only continuation pass is scaffolded** (commit `dd7b202`) — `configs/pilot_250m_decay.yaml` + `--reset-data-position-on-resume` CLI flag + eval-hook int32 fix. **NOT YET LAUNCHED.** Decision: ship-as-is vs run-decay-pass is open.
- **All artifacts on R2** at `s3://llm-data/`. Final checkpoint at `s3://llm-data/checkpoints/pilot-250m-v1/step-000151990/`. eval-final.json at `/workspace/eval-final.json` on the pod (NOT yet uploaded to R2).
- **Two W&B runs** for the pilot due to the int32 crash + resume: `roydqofb` (pre-crash, steps 0–65,500) and `u5xsxm0l` (post-resume, steps 65,000–151,990). Both finalized.
- **Three production bugs fixed during the run**: int32 data_position overflow in train loop (`9f442f7`), int32 in eval-hook path (`dd7b202`), the eval-on-resume silent failure (same `dd7b202`).
- **Critical Stage 2/3 finding**: pilot corpus is single-epoch only at this batch size. For runs > 152K steps at mb=4, we need multi-epoch iteration or a bigger corpus.

If you're picking this up cold, you only need:
1. Read this doc (~15 min — §0, §1, §13 in particular)
2. Glance at `docs/PROJECT_OVERVIEW.md` for project canon (~10 min)
3. Glance at W&B runs (`roydqofb` + `u5xsxm0l`) for the curves

---

## 1. Where the live pilot stands

### Run identity
- **W&B run**: `pilot-250m-v1-2026-05-13` (URL: https://wandb.ai/harshit-hvpals-ahurasense/myllm/runs/roydqofb)
- **Pod**: 4×H200 SXM on RunPod (`pilot` tmux session)
- **Code**: `main` at commit `9f442f7` (post int32 fix)
- **Start time**: 2026-05-14T00:03:38Z
- **Config**: `configs/pilot_250m.yaml` + `configs/data/pretrain_mix_pilot.yaml`
- **Corpus**: `/workspace/corpus_pilot_train/` (pulled from `s3://llm-data/corpus_v1_pilot/train/`)
- **Tokenizer**: `artifacts/tokenizer_v1.json` (production v1 SPM-Unigram, sha `0ad881f58dab…`)
- **Local checkpoint dir**: `/workspace/ckpt/pilot-250m-v1/`
- **R2 checkpoint prefix**: `s3://llm-data/checkpoints/pilot-250m-v1/`

### Launch command (verbatim, for resume)
```bash
python scripts/run_pretrain.py \
    --model-config configs/pilot_250m.yaml \
    --data-config configs/data/pretrain_mix_pilot.yaml \
    --tokenizer-path artifacts/tokenizer_v1.json \
    --packed-corpus-root /workspace/corpus_pilot_train \
    --run-name pilot-250m-v1-2026-05-13 \
    --total-steps 229000 \
    --micro-batch-override 4 \
    --log-every 100 \
    --eval-every 5000 \
    --eval-n-batches 32 \
    --checkpoint-every 5000 \
    --checkpoint-root /workspace/ckpt/pilot-250m-v1 \
    --checkpoint-r2-prefix checkpoints/pilot-250m-v1 \
    2>&1 | tee -a /workspace/pilot.log
```

### Pod env vars that must be set (run from `~/.bashrc`)
```
AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET=llm-data,
S3_ENDPOINT_URL (cloudflare R2), AWS_DEFAULT_REGION=auto,
HF_TOKEN, WANDB_API_KEY,
NCCL_NVLS_ENABLE=0, NCCL_IB_DISABLE=1
```
Plus `unset LD_LIBRARY_PATH` (RunPod images pre-set this to shadow JAX's bundled cuDNN).

### Progress curve (approximate, smoothed)
```
step      |  loss  | note
─────────────────────────────────────
   100    | 11.04  | random init (log(131072)=11.78)
 5,000    |  ~4.0  | first checkpoint
10,000    |  ~3.4  |
20,000    |  ~3.0  | warmup well behind, NaN rate dropped
50,000    |  ~2.9  |
67,500    |  2.66  | best so far
65,500    |  CRASH | int32 overflow of data_position (FIXED, see §4)
75,000    |  2.65  | post-resume, settling
77,000    |  2.80  | noisier but progressing
80,000    |  ~2.7  | latest checkpoint on R2 (step 80K)
```
Final loss target: 2.3–2.5 by step 229,000.

### NaN burnout (verified healthy)
142 NaN events over 67,126 steps = 2.1/1000 — under the 5/1000 threshold.
Source distribution proportional to corpus shares → **bf16 numerical sensitivity, not bad data.** No `hard_spike` events ever fired; atomic NaN-skip handled every event.

Run `python scripts/inspect_quarantine.py /workspace/ckpt/pilot-250m-v1/quarantine.jsonl --corpus-root /workspace/corpus_pilot_train` for live numbers.

---

## 2. What got done in this session (2026-05-13 → 2026-05-14)

In rough order, what happened:

1. **Corpus build complete** — 5B tokens across 13 sources via `run_parallel_builds.py`. Wall time 2h 1m on the 128-core dev box. All sources uploaded to `s3://llm-data/corpus_v1_pilot/sources/`.
2. **Compose pass** — `scripts/compose_mixed_corpus.py --strict-sources` interleaved the 13 per-source corpora into a single training corpus at `s3://llm-data/corpus_v1_pilot/train/`. 608,088 sequences = 4.98B tokens, 10 shards, ~20 GB. Max source-share drift 0.33%, L5 verify PASS.
3. **Pilot dry-run on synthetic** — 5 steps on dev-box CPU validated the full plumbing (config load, model build, optimizer, eval hook, checkpoint save). Loss 11.81x at random init — sane.
4. **Watchdog stress test** — 5/5 existing pytest tests in `tests/test_watchdog_recovery.py` PASS (validates the atomic NaN-revert + LR halving + batch-skip rollback path under controlled spike injection).
5. **Pod setup on 4×H200 SXM** — venv install, NCCL_NVLS workaround discovered + fixed (intra-node NVLink SHARP fails on RunPod without fabric manager).
6. **Pilot launched** at 2026-05-14T00:03:38Z. W&B `roydqofb`.
7. **Mid-run crash at step ~65,500** — `data_position` overflowed int32 (~2^31 tokens = 2.147B). Fixed at `src/myllm/training/loop.py:193` (commit `9f442f7`): pop data_position before train_step_fn (it doesn't use it inside the JIT'd call), restore as Python int after. No JAX retyping.
8. **Resumed cleanly** from step-65000 R2 checkpoint. Training continues; step 80K checkpoint mirrored to R2.
9. **Forensic tool added** — `scripts/inspect_quarantine.py` reads `quarantine.jsonl` and maps each NaN-skip's `data_position` back to the source-mix that contributed to that packed sequence. Reveals source-attribution to detect poisonous data vs bf16 numerical noise. Used twice today; confirmed bf16 hypothesis.
10. **Rust migration v0.2.1 plan locked** at `docs/stage3_rust_migration_plan.md`. External reviewer found 3 material flaws + 2 simplifications in the v0.1 plan, all folded in. Implementer working on a separate fork (`cs2hvh/llm-build-rust`, branch `main`).
11. **dev-small branch** created with a 50M mini-pilot config for lifecycle validation. NOT yet run.
12. **8×H200 SXM pod brought up** in parallel (separate procurement) for Stage 2 prep. Setup partial — venv built, corpus pulled. Not yet used.

---

## 3. Project architecture (one-paragraph + one-diagram)

**MyLLM is a multi-component pretraining stack with strict separation between data construction, training, and eval.** Each artifact (corpus, tokenizer, checkpoints, eval scorecard) is persisted to R2 as the durable truth; local filesystem is ephemeral. Training runs on rented GPU pods (RunPod) that pull artifacts from R2, train, and mirror state back. The dev box (128-core CPU server with .env credentials) is the control plane.

```
                   ┌────────────────────────────┐
                   │     Cloudflare R2 (canon)   │
                   │  s3://llm-data/             │
                   │   ├── tokenizer/            │
                   │   ├── decontamination/      │
                   │   ├── corpus_v1_pilot/      │
                   │   │    ├── sources/...      │
                   │   │    └── train/...        │
                   │   └── checkpoints/          │
                   │        └── pilot-250m-v1/   │
                   └─────────┬──────────────────┘
                             │ R2 multipart I/O (boto3, 32MB chunks, adaptive retry)
              ┌──────────────┼──────────────────┐
              │              │                  │
   ┌──────────▼──────┐  ┌────▼──────────┐  ┌───▼────────────┐
   │ DEV BOX (control │  │ GPU POD       │  │ EVAL/RELEASE   │
   │ plane, 128-core) │  │ (4×H200, etc) │  │ (any GPU pod)  │
   │                  │  │               │  │                │
   │  - corpus build  │  │  - run_pretrain.py            │  - build_release_scorecard.py
   │  - compose       │  │  - JAX + Keras 3 + Orbax │  - benchmarks  │
   │  - tokenizer train│  │  - watchdog + NaN-skip    │  - greedy decode │
   │  - .env source   │  │  - WSD LR schedule        │  - JSON+MD output│
   └──────────────────┘  └───────────────┘  └────────────────┘
```

### Key code modules (so you can navigate fast)

```
src/myllm/
├── model/
│   ├── transformer.py       ← Llama-style decoder, GQA, RoPE, RMSNorm, SwiGLU
│   ├── config.py            ← Pydantic ModelConfig + MupConfig
│   └── layers.py            ← DecoderBlock, GroupedQueryAttention, etc.
├── training/
│   ├── train_step.py        ← jit'd forward + backward + atomic NaN-revert
│   ├── loop.py              ← train_loop driver (watchdog, ckpt cadence, eval-hook plug-in)
│   ├── optimizer.py         ← AdamW + muP multi_transform + fp32-moments pin
│   ├── checkpoint.py        ← Orbax wrapper, R2 mirror (G6 reshard fix STILL PENDING)
│   ├── eval_hook.py         ← --eval-every machinery (FSDP-incompatible currently)
│   ├── decay_phase.py       ← teacher-distillation injection (Stage 3 feature, dormant during pilot)
│   ├── watchdog.py          ← loss-spike detector (3σ soft, 6σ hard)
│   ├── quarantine.py        ← bad-batch dumper (writes quarantine.jsonl)
│   └── schedule.py          ← WSD LR schedule + decay_steps resolver
├── data/
│   ├── packed_corpus.py     ← PackedCorpusWriter + PackedCorpusReader (LOCAL ONLY; no S3 streaming yet)
│   ├── compose.py           ← multi-source deficit-driven sampler
│   ├── build.py             ← per-source pipeline (filter → tokenize → pack → R2)
│   ├── decontamination.py   ← MinHash dual-mode 8+13gram
│   └── filters.py           ← length, repetition, symbol-ratio, PII
└── eval/
    ├── release_scorecard.py ← Scorecard data class + format machinery (predict_fn STUB)
    └── benchmarks/          ← per-benchmark adapters

scripts/
├── run_pretrain.py          ← main entry point for any training run
├── build_packed_corpus.py   ← per-source corpus builder
├── compose_mixed_corpus.py  ← compose pass driver
├── run_parallel_builds.py   ← orchestrator for fan-out across sources
├── canary_ladder.py         ← L0/L5 sanity checks
├── inspect_quarantine.py    ← forensic NaN-skip tool (added today)
├── build_release_scorecard.py  ← scorecard CLI (predict_fn STUB)
├── pod_setup_apt.sh         ← pod system pkgs
└── pod_launch_gpu.sh        ← canonical pip stack (torch 2.7.1 + jax[cuda12] 0.4.38)
```

### Architectural locked decisions
- **Backend**: Keras 3 with JAX backend. JAX is the math; Keras handles layer composition.
- **Precision**: bf16 forward + backward; fp32 master copy + Adam moments pinned to fp32.
- **Sharding**: pure DP (data-parallel replicated) for 250M pilot. FSDP/ZeRO-3 required for 1B+ at ctx=8192 (Stage 2+).
- **Optimizer**: AdamW + Optax `multi_transform` for muP per-group LR scaling.
- **LR schedule**: WSD (Warmup-Stable-Decay) — SmolLM2 / MiniCPM standard.
- **Hash**: xxh64 for MinHash signatures (NOT rensa — incompatible with existing R2 decontam indexes; see Rust plan D3).
- **Corpus order**: deficit-driven sampling for compose, preserves target source shares within 2% (L5 gate).
- **Checkpoint**: Orbax sharded, R2 mirror per save. data_position + step + opt_state all preserved for bit-exact resume.

---

## 4. Critical gotchas / known issues

### Solved (don't re-litigate)
- **int32 overflow of `data_position`** → FIXED in `9f442f7`. data_position is now popped before train_step and restored as Python int.
- **NCCL_NVLS init failure on RunPod H200** → set `NCCL_NVLS_ENABLE=0` + `NCCL_IB_DISABLE=1`. Permanent in `~/.bashrc` on pod.
- **bf16 NaN sensitivity** → ~2-3 NaN-skips per 1000 steps is normal. Source attribution confirms no single bad source. Atomic skip handles each event.
- **uint16 → uint32 token storage** → token IDs need uint32 on disk because 131k vocab > 65535 (uint16 max).
- **Mistral / Qwen3.6 dropped as teachers** → license verification on 2026-05-12 caught restrictions. Teacher plan v2 locked to DeepSeek-V4-Pro-Base + Olmo-3-32B-Base.
- **Spotty R2 downloads of large shards** → use boto3 with `Config(retries={'max_attempts': 10, 'mode': 'adaptive'})` + 32 MB chunks. CLI `aws s3 sync` flakes on large files.

### Known unfixed
- **G6 reshard bug at `src/myllm/training/checkpoint.py:143`** → `orbax.restore()` lacks `RestoreArgs(sharding=...)`. Can't resume a checkpoint saved on DP=N onto DP=M. Blocks cross-mesh runtime resume. Required before Stage 2 pod-shape flexibility.
- **R2 connection pool spam during checkpoint upload** → boto3 default pool size 10 is too small for Orbax's ~30+ concurrent shard uploads. Cosmetic but noisy. Fix: bump `max_pool_connections=50` in `src/myllm/utils/storage.py`. Pending todo.
- **PackedCorpusReader doesn't accept S3 URIs** → must pre-download corpus to local. Annoying for ephemeral pods. Stage 2 prep todo.
- **`predict_fn` in `release_scorecard.py` is a NotImplementedError stub** → can't actually score benchmarks against a real checkpoint yet. Pending implementation; ~4-6 hr engineering.
- **`--eval-every` disabled under FSDP** → `donate_argnums=(0,)` corrupts state on forward-only eval re-run. Stage 2 prep (forward-only eval_step refactor).
- **`pretrain_mix.yaml` line 75 has stale `bigcode/starcoderdata`** → real corpus uses codeparrot fallback (gated source replaced). Edit before next compose if using `pretrain_mix.yaml` (NOT `pretrain_mix_pilot.yaml` — that one is correct).
- **CLI `--micro-batch-override` is GLOBAL batch size, not per-device** → must be divisible by DP. 4-GPU pod uses mb=4, 8-GPU pod needs mb=8 minimum.

### Operational warnings
- **CUDA 12.4 ptxas miscompile warning** → harmless for our workload; H200 pods print it on every JIT but no known bug triggered.
- **File-deletion gremlin** → `docs/CLAUDE_COLLAB.md` and `docs/review/QUERIES_FOR_REVIEWER_2026-05-12-evening.md` have intermittently disappeared from working tree. Restored with `git checkout HEAD -- <files>`. Root cause unknown; doesn't affect committed state.

---

## 5. Open todos (prioritized by phase)

### A. Right now (pilot is running) — no action needed
- Wait for pilot completion (~04:20 UTC 2026-05-15)
- Monitor W&B + R2 checkpoint cadence

### B. Within 1-2 hours after pilot completes
1. **Verify `training_complete` event** in `/workspace/pilot.log`
2. **Verify final checkpoint at step-229000 on R2**
3. **Run release scorecard** with mock predict (validates pipeline machinery): `python scripts/build_release_scorecard.py --checkpoint .../step-000229000 --use-mock-predict ...`
4. **Decide: implement real predict_fn?** — ~4-6 hr engineering. Necessary for real benchmark scores. Currently a `NotImplementedError`.

### C. Before Stage 2 launch (3-5 days engineering)
| Item | File | Effort |
|---|---|---|
| P0-1: Per-source val loss in eval hook | `src/myllm/training/eval_hook.py` + `run_pretrain.py` | 10-14 hr |
| P0-2: `--production` flag + fail-closed packed-corpus check | `scripts/run_pretrain.py` | 2-3 hr |
| P0-3: Packed-resume fail-closed invariant | `src/myllm/data/packed_corpus.py::peek_data_position_from_checkpoint` | 2-3 hr |
| G6 reshard fix (orbax) | `src/myllm/training/checkpoint.py:143` | 2-3 hr |
| Forward-only eval_step for FSDP | `src/myllm/training/eval_hook.py` | 4-6 hr |
| R2 client `max_pool_connections=50+` | `src/myllm/utils/storage.py` | 30 min |
| S3 streaming in PackedCorpusReader | `src/myllm/data/packed_corpus.py` | 6-8 hr |

### D. Polish / nice-to-have (no specific gate)
- `tests/test_docs_config_sync.py` — catch config-doc drift (Session B prepped, never landed)
- Fix RoPE doc drift in `docs/PROJECT_OVERVIEW.md:271,311` (says 130000, config says 500000)
- Fix `scripts/run_parallel_builds.py` summary parser (cosmetic — "could not parse summary" warnings on healthy runs)
- Investigate HF-datasets teardown SIGABRT on fineweb-edu (cosmetic)

### E. Stage 3 prep
- Real-text teacher audit re-run (decides K for distillation cache)
- Teacher logit caching (DeepSeek-V4-Pro-Base + Olmo-3-32B-Base forward passes; ~22 TB on R2)
- Compose v2 / 1B corpus build (larger per-source budgets, possibly bigger than pilot)

### F. Parallel work (separate session/owner)
- **Rust migration** — implementer working on `cs2hvh/llm-build-rust` fork (separate GitHub repo). v0.2.1 plan locked at `docs/stage3_rust_migration_plan.md`. Phase 0+ (pre-Rust Python wins) is the immediate next step. See plan §4 for phase breakdown.

### G. User action items
- **HF access for ai4bharat/MILU**: harshit.hv@samatva.com must request at https://huggingface.co/datasets/ai4bharat/MILU. Blocks MILU addition to decontam index + eval suite.

---

## 6. Next plans — phase by phase

### Phase A — Pilot wraps (auto, ~15 hours from now)
1. Pilot reaches step 229,000 → `training_complete` event
2. Final checkpoint mirrored to R2
3. W&B run finalizes

### Phase B — Pilot post-mortem (1-2 days)
1. Run release scorecard with real predict_fn → benchmark scores on MMLU-pro, GSM8K, HumanEval-plus, IFEval, Belebele
2. Update `docs/governance/model_card_v1.md` with real numbers
3. Memorization probe: sample 100 prompts, check for verbatim training data
4. NaN-rate / loss-curve analysis writeup
5. Friend reviewer pass: did the pilot validate the recipe?

### Phase C — Go/no-go for Stage 2
**Decision criteria:**
- Pilot completed cleanly with `training_complete`
- Final loss in expected 2.3-2.5 range
- Benchmark scores not catastrophic (e.g., HellaSwag > random)
- No discovered correctness gaps
- Friend reviewer signs off

If GO → Phase D. If issues → fix and either re-pilot or fix-forward.

### Phase D — Stage 2 prep (3-5 days)
Land all "Before Stage 2" items from §5C above. Especially:
- P0-1 (per-source val loss) — gives diagnostic visibility during the long Stage 2/3 runs
- G6 reshard fix — unlocks cross-mesh runtime resume
- Forward-only eval_step — lets `--eval-every` work under FSDP

### Phase E — Stage 2 launch (3-5 days wall)
**1B model, 10-30B tokens, FSDP-required.**
- Config: `configs/base_1b.yaml`
- Pod: 4× B200 (recommended, per `feedback_h200_throughput_baseline`) OR 8× H200 SXM
- Cost: $700-2000
- Goal: validate the recipe scales 250M → 1B. If Stage 2 trains stable, the recipe is real.

### Phase F — Stage 3 prep (1-2 weeks)
- Real-text teacher audit re-run
- Teacher logit caching (DeepSeek-V4-Pro-Base + Olmo-3-32B-Base; ~22 TB on R2)
- Compose v2 corpus (~200-600B tokens depending on final budget)
- B200 pod procurement for ~30-day run

### Phase G — Stage 3 base v1 (~30 days wall, ~$13K)
**1B model @ 600B tokens.** The real deliverable.

### Phase H — Release (~1 week)
- Final scorecard, model card, data card, license register
- Memorization probes
- Safety policy audit
- Public release decision

---

## 7. R2 inventory (verified 2026-05-14 13:00 UTC)

```
s3://llm-data/                       (Cloudflare R2, "auto" region)
├── tokenizer/
│   └── myllm-spm-unigram-131k-v2.json    4.79 MB    production v1, sha 0ad881f58dab
├── decontamination/
│   ├── decontamination_index_8gram.json  37.52 MB   10 benchmarks, 1.75M ngrams
│   └── decontamination_index_13gram.json 37.21 MB   10 benchmarks, 1.74M ngrams
├── corpus_v1/                       OLD pre-2026-05-13 builds at seq_len=8192 (WRONG; off-by-one)
├── corpus_v1_pilot/                 CURRENT pilot corpus at seq_len=8193
│   ├── sources/<source-id>/         per-source per-shard
│   └── train/                       composed corpus, 10 shards, ~20 GB, 4.98B tokens
├── checkpoints/
│   └── pilot-250m-v1/
│       ├── step-000005000/   ← every 5000 steps
│       ├── step-000010000/
│       ├── ...
│       └── step-000080000/   ← latest as of 2026-05-14T12:38:32Z, ~2.65 GB each
└── (other prefixes: code/, corpus_pilot/, fsdp_gauntlet/, teacher_audit/)
```

Total R2 usage right now: ~70 GB (20 GB corpus + 42 GB checkpoints + small artifacts).

---

## 8. Auto-memory (persistent across Claude sessions)

Memory files at `/root/.claude/projects/-root/memory/`:

- `user_role.md` — solo lead, enterprise stance
- `feedback_enterprise_rigor.md` — root-cause fixes preferred
- `feedback_control_plane_pattern.md` — ephemeral workers, R2 as durable truth
- `feedback_verify_before_locking.md` — verify external claims (we got burned on teacher licenses)
- `reference_external_resources.md` — R2 bucket, HF, .env state
- `feedback_tokenizer_scale_ceiling.md` — use native SentencePiece (HF Unigram hits ~15GB ceiling)
- `project_teacher_strategy.md` — DeepSeek-V4-Pro-Base + Olmo-3-32B-Base (locked 2026-05-12)
- `feedback_uint32_for_131k_vocab.md` — token IDs must be uint32 (silent overflow risk at uint16)
- `feedback_h200_throughput_baseline.md` — 280-360K tok/sec aggregate on 8×H200
- `feedback_no_force_reinstall.md` — never `--force-reinstall` on nvidia-* extras
- `reference_canonical_gpu_pin.md` — torch 2.7.1 + jax[cuda12] 0.4.38
- `project_fsdp_validated_2026-05-13.md` — G1-G4 PASS on 2×H200 SXM (G5 op, G6 broken)
- `project_session_state_2026-05-13.md` — yesterday's resume pointer (now superseded by THIS file)
- `project_rust_migration_v0_2.md` — Rust migration v0.2/v0.2.1 locked decisions

---

## 9. Key commits worth orienting against

| Hash | Subject |
|---|---|
| `9f442f7` | **loop: fix int32 overflow of data_position at ~2.1B tokens** (this morning, the live bug) |
| `951b2cc` | inspect_quarantine v0.2: read composed-corpus seq_meta schema correctly |
| `dfabe11` | inspect_quarantine: forensic tool for NaN-skip post-mortems |
| `c720564` | Rust migration v0.2.1: cleanup of stale upstream refs |
| `a39e665` | Rust migration v0.2: external reviewer pass |
| `8ab94a1` | dev-small: 100M mini-pilot config (now stale; renamed to 50M, but config edit not pushed) |
| `fb6a537` | release scorecard scaffold (predict_fn STUB) |
| `518aa50` | eval-during-training: --eval-every wires val_loss + perplexity |
| `7331377` | pilot: D2 — bump pilot ctx 4096→8192 |
| `03a8b3a` | pilot corpus plan: 5B-token rebuild + pretrain_mix_pilot.yaml (codeparrot swap) |
| `6cf299e` | canonical single-venv recipe: torch==2.7.1 + jax[cuda12]==0.4.38 |
| `cc56daa` | Decontam: wire DualModeDecontaminationIndex through corpus build |

---

## 10. Files to read for fuller context

In rough order of "if you want to know more about X, read Y":

| Topic | File |
|---|---|
| Project canon | `docs/PROJECT_OVERVIEW.md` (800 lines, kept current) |
| Friend-reviewer-facing status | `docs/review/STATUS_2026-05-13.md` |
| Pilot corpus rebuild rationale | `docs/pilot_corpus_rebuild_plan.md` |
| Yesterday's session pointer | `docs/SESSION_STATE_2026-05-13.md` (untracked, may not be on this branch) |
| Stage 3 Rust migration plan | `docs/stage3_rust_migration_plan.md` (v0.2.1 LOCKED) |
| muP design + scaling rules | `docs/mup_design.md` |
| Teacher distillation strategy | `docs/teacher_distillation_strategy.md` |
| Teacher logit cache format | `docs/teacher_logit_cache_format.md` |
| Math strategy | `docs/math_strategy.md` |
| Playbook alignment | `docs/playbook_alignment.md` |
| Safety policy | `docs/safety_policy.md` |
| Model card v1 | `docs/governance/model_card_v1.md` |
| Data card v1 | `docs/governance/data_card_v1.md` |
| License register | `docs/governance/license_register.md` |
| Pod bring-up recipe | `scripts/pod_launch_gpu.sh` |
| Quarantine forensic tool | `scripts/inspect_quarantine.py` |

---

## 11. How to resume / verify / restart if pod dies

### Pilot is alive and you want to check on it
```bash
ssh ...pod...
tmux attach -t pilot
# Loss + step events stream by. Detach: Ctrl-B then D.
```

### Pilot crashed or pod died — resume from latest R2 checkpoint
1. SSH to a fresh pod (or restart same pod). Set env vars (see §1).
2. `git clone https://github.com/cs2hvh/ai-llm.git /workspace/llm-build && cd $_`
3. `bash scripts/pod_setup_apt.sh && bash scripts/pod_launch_gpu.sh && source .venv/bin/activate`
4. Pull tokenizer + corpus from R2 (see §1).
5. **Re-launch with same args** — `--checkpoint-root /workspace/ckpt/pilot-250m-v1` auto-resumes from the latest checkpoint at that path (if the pod's `/workspace` is fresh, the checkpoint will be downloaded from R2 first; actually, current behavior requires the local ckpt dir to exist; if not, you may need to manually `aws s3 sync` the latest step-* dir down before re-launching).

### Pilot finished cleanly
1. Verify: `grep '"event": "training_complete"' /workspace/pilot.log`
2. Confirm final ckpt on R2: `aws s3 ls s3://$S3_BUCKET/checkpoints/pilot-250m-v1/step-000229000/`
3. Run release scorecard (see §6 Phase B)

---

## 12. Contact / accountability

- **User**: harshit.hv@samatva.com (solo lead)
- **GitHub**: cs2hvh/ai-llm (main repo), cs2hvh/llm-build-rust (Rust migration fork)
- **W&B project**: harshit-hvpals-ahurasense/myllm
- **Cloudflare R2 bucket**: llm-data (account-specific endpoint in .env)
- **Friend reviewer**: external, async via the `docs/review/` packets

When the next session opens, your first action should be reading this doc + `docs/PROJECT_OVERVIEW.md`, then glancing at the W&B run. If you have questions a doc doesn't answer, the user's auto-memory at `/root/.claude/projects/-root/memory/` will likely have the call-pattern history.

---

## 13. POST-PILOT UPDATE (2026-05-14 ~22:00 UTC)

This section captures what happened AFTER §1–§12 were written. Read this for the current state of the project.

### 13.1 Pilot completion — corpus exhausted early

The pilot completed at **step 151,990** instead of the planned 229,000. Cause:

```
Corpus: 608,088 sequences × 8192 tokens = 4.98 B tokens
mb_global = 4 (--micro-batch-override 4) → 4 sequences per step
Steps to exhaust corpus = 608,088 / 4 = 152,022
```

The data iterator hit end-of-corpus at step 151,990 (32 steps before the calculated exhaustion point — minor accounting for the eval hook's held-out batches), and `training_complete` fired cleanly with `final_step: 151990`. No crash, just an early-stop on data exhaustion.

**Implications:**
- **WSD decay phase never reached.** Decay was scheduled at `total_steps × (1 - decay_fraction) = 229000 × 0.85 = 194,650`. The model trained at full peak LR (3e-4) for the entire stable phase, never getting the "model settles into local minimum" benefit.
- **Final train loss**: smoothed ~2.4 (last 100 steps showed 2.11–2.73 oscillation).
- **NaN events**: 288 total, 1.9/1000 rate — healthy. Watchdog never triggered hard rollback; `lr_recovery_multiplier=1.0` throughout.
- **W&B run split**: the int32 crash + resume at step ~65K created a SECOND W&B run (`u5xsxm0l`) instead of continuing `roydqofb`. Both are finalized; the curves are split across them.

### 13.2 Post-hoc eval results (commit `70b9009`)

Built `scripts/eval_checkpoint.py` to compute val_loss/val_ppl on any saved checkpoint without re-launching training. Loads weights from Orbax checkpoint, iterates the first N batches of the corpus (the original held-out set), computes mean CE + perplexity.

Ran against the final checkpoint (step 151,990):

```
==============================================
checkpoint   : step-000151990
n_batches    : 32 (matches the in-training hook's held-out set)
micro_batch  : 4
val_loss     : 2.877632
val_ppl      : 17.7721
==============================================
```

**Comparison to in-training eval points** (pre-crash, last 3):
- step 55,000: val_loss=2.951, val_ppl=19.13
- step 60,000: val_loss=2.992, val_ppl=19.93
- step 65,000: val_loss=2.975, val_ppl=19.59
- **step 151,990: val_loss=2.878, val_ppl=17.77** ← post-hoc

So between step 65K (last in-training eval) and step 152K, val_loss improved by **0.10 nats / val_ppl reduced 9%**. Train-val gap is 0.48 nats (smoothed train 2.4 vs val 2.88) — borderline acceptable for a pilot. The eval result has been written to `/workspace/eval-final.json` on the pod but **NOT yet uploaded to R2** (todo).

### 13.3 Bugs discovered + fixed during the run

**Bug 1: int32 overflow of `data_position` in train_step** (fixed in commit `9f442f7`)
- `state["data_position"]` exceeds `2^31 = 2,147,483,648` at step ~65,536 (mb=4 × seq=8192 = 32,768 tokens/step). JAX default-types Python ints as int32 when tracing inputs to JIT'd functions. Crash:
  ```
  OverflowError: Python int 2147483648 too large to convert to int32
  ```
- Fix at `src/myllm/training/loop.py:193`: pop `data_position` from a shallow state copy before each `train_step_fn` call, restore as Python int after. data_position is not used INSIDE train_step (just carried through state for checkpointing).

**Bug 2: same int32 issue in the eval hook** (fixed in commit `dd7b202`)
- `src/myllm/training/eval_hook.py:make_validation_loss_eval` re-uses `train_step_fn` for eval but didn't apply the same pop. Result: post-resume evals silently failed (`eval_failed_non_fatal` warnings; no `eval` events).
- Fix at `src/myllm/training/eval_hook.py:75-85`: same shallow-copy-without-data_position pattern.

**Bug 3: silent eval failure post-resume**
- Compound effect of Bug 2: the in-training eval hook fired at the expected intervals but always raised, caught by the loop's try/except, logged as `eval_failed_non_fatal` warnings. No `eval` events appeared in the log past step 65,000.
- That's why `tail -3` of the `eval` events shows only steps 55K/60K/65K — every post-resume eval was silently broken.

### 13.4 Stage 1.5 decay-only pass — SCAFFOLDED (commit `dd7b202`), launch pending

The pilot result is borderline under-converged because WSD decay never ran. A Stage 1.5 continuation pass loads the step-151,990 checkpoint, rewinds the data cursor, and runs ~20K more steps at a decaying LR (3e-4 → 3e-5).

**Files added/changed in commit `dd7b202`:**
- `src/myllm/training/loop.py` — `LoopConfig.reset_data_position_on_resume: bool = False`. When `True`, the restore code in `run()` rewinds `state["data_position"]` to 0 after loading the checkpoint. Logs `data_position_reset_on_resume` warning.
- `scripts/run_pretrain.py` — new CLI flag `--reset-data-position-on-resume`. Wired both to LoopConfig AND to the packed-corpus iterator's `start_sequence_id` resolution (both need to reset; otherwise iterator and state disagree on next checkpoint).
- `src/myllm/training/eval_hook.py` — int32 bug fix (Bug 2 above).
- `configs/pilot_250m_decay.yaml` — same architecture as pilot_250m.yaml but with `lr_schedule.warmup_steps: 0` and `lr_schedule.decay_fraction: 0.1163`. With `--total-steps 171990`, decay phase covers steps 151,990 → 171,990.

**Schedule math** (so the next session can verify):
- total_steps = 171,990
- decay_fraction = 20000 / 171990 = 0.1163
- decay starts at: total_steps × (1 − decay_fraction) = 171,990 × 0.8837 = 151,990 ✓
- decay window: 20,000 steps from step 151,990 to 171,990
- LR walks linearly from peak_lr=3e-4 down to peak_lr × end_lr_ratio = 3e-5

**Launch command (NOT YET RUN):**
```bash
# Pre-stage the checkpoint into a fresh dir so we don't overwrite the original
mkdir -p /workspace/ckpt/pilot-250m-v1-decay
cp -r /workspace/ckpt/pilot-250m-v1/step-000151990 /workspace/ckpt/pilot-250m-v1-decay/

# Launch in tmux
tmux new -s decay
python scripts/run_pretrain.py \
    --model-config configs/pilot_250m_decay.yaml \
    --data-config configs/data/pretrain_mix_pilot.yaml \
    --tokenizer-path artifacts/tokenizer_v1.json \
    --packed-corpus-root /workspace/corpus_pilot_train \
    --run-name pilot-250m-v1-decay-2026-05-14 \
    --total-steps 171990 \
    --micro-batch-override 4 \
    --log-every 100 \
    --eval-every 1000 \
    --eval-n-batches 32 \
    --checkpoint-every 2000 \
    --checkpoint-root /workspace/ckpt/pilot-250m-v1-decay \
    --checkpoint-r2-prefix checkpoints/pilot-250m-v1-decay \
    --reset-data-position-on-resume \
    2>&1 | tee /workspace/pilot-decay.log
```

**Cost / wall**: ~2 hours, ~$30 on 4×H200 SXM.

**Expected outcome** (calibrated honestly after literature recheck):
- val_loss after decay: realistically **2.65–2.75** (NOT 2.4–2.5 — I was optimistic in my first projection to the user)
- val_ppl after decay: ~14–16
- Improvement: 0.13–0.23 nats vs current 2.878

The primary motivation for running Stage 1.5 is NOT the loss number (small gain). The real reasons:
1. **Validates the WSD decay code path** — currently zero production runtime confirmation. Stage 3's $13K base run depends on this code working.
2. **Validates the eval-hook int32 fix** — committed as a code change but never executed in production.
3. **Validates the `--reset-data-position-on-resume` flag** — same.

User has been informed of the honest expected outcome. **Decision is open** — either run Stage 1.5 (~$30) or ship-as-is.

### 13.5 Stage 2/3 corpus capacity finding — CRITICAL FOR PLANNING

The pilot's "training stopped early" surprise revealed a real planning issue:

```
Pilot corpus: 4.98 B tokens, 608K sequences at seq_len=8192
mb_global = 4 → 1 sequence per device per step (DP=4)
Max steps before corpus exhaustion = 152K
Stage 3 target: 600 B tokens / 30 days base run = ~18M steps at this batch size
```

**The current `PackedCorpusReader` is single-epoch.** When it hits end-of-corpus, the loop exits with `training_complete`. There's no looping back to start.

For Stage 2 (1B model rehearsal at 10–30B tokens):
- 30B tokens / 32,768 tokens-per-step = 916K steps. **6× the pilot corpus.** Need either a much bigger Stage 2 corpus OR multi-epoch iteration.

For Stage 3 (1B base at 600B tokens):
- 600B tokens / 32,768 = 18.3M steps. **120× the pilot corpus.** Need a much larger v2 corpus build (not the pilot corpus).

**Action required before Stage 2 launch:**
- Option A: build a larger Stage 2 corpus (~30B+ tokens). Substantial — would re-build all 13 sources at higher per-source targets.
- Option B: add multi-epoch support to `PackedCorpusReader`. Easier (a couple of hours engineering) but model sees each doc multiple times.
- Option C: Stage 2 rehearsal accepts single-epoch and is sized to the available corpus (~5B tokens for 250M, or proportionally larger for 1B).

Recommendation: Option B is the right path. Multi-epoch is standard practice and pre-Chinchilla wasn't an issue. With our small corpus, we'd typically see each doc 2-4 times anyway.

**This belongs in the Stage 2 prep todos**, joining P0-1, P0-2, P0-3, G6 reshard, FSDP eval.

### 13.6 Recent commits (between the previous handoff and now)

| Hash | Subject |
|---|---|
| `dd7b202` | **Stage 1.5: decay-only continuation pass scaffolding** |
| `70b9009` | scripts/eval_checkpoint.py — post-hoc val_loss/val_ppl from any checkpoint |
| `9f442f7` | loop: fix int32 overflow of data_position at ~2.1B tokens |

All on `main`, all pushed to `origin/main`.

### 13.7 Updated open todos (as of 22:00 UTC)

Critical path:
1. **Decide: run Stage 1.5 or ship pilot as-is** (USER) — affects model card writeup
2. **If Stage 1.5**: launch via the command in §13.4, wait ~2 hr, run `eval_checkpoint.py` against the post-decay checkpoint
3. **Upload `/workspace/eval-final.json` to R2** (small, ~1 KB; add to `s3://llm-data/checkpoints/pilot-250m-v1/step-000151990/`)

Stage 2 prep (in addition to existing items):
4. **Multi-epoch support in PackedCorpusReader** (Option B above) — Stage 2 blocking unless we accept Option C
5. Original Stage 2 prep items: P0-1 (per-source val loss), P0-2 (production flag), P0-3 (packed-resume safety), G6 reshard, forward-only eval_step (Bug 2 from §13.3 superseded part of this)

Polish:
6. Bump R2 `max_pool_connections=50+` (still pending; cosmetic checkpoint-upload spam)
7. Update `docs/PROJECT_OVERVIEW.md` to reflect pilot completion + val_loss number
8. Update `docs/governance/model_card_v1.md` with real numbers (pending Stage 1.5 decision)
9. Write release scorecard `predict_fn` (currently NotImplementedError stub)
10. `tests/test_docs_config_sync.py` + RoPE drift fix
11. `S3 streaming for PackedCorpusReader`

### 13.8 Where the pod state stands

The 4×H200 SXM pod from the pilot is still up as of writing. Has:
- `/workspace/llm-build/` with code at commit `dd7b202` (after `git pull`)
- `/workspace/corpus_pilot_train/` with the 20 GB corpus (verified 10/10 shards)
- `/workspace/ckpt/pilot-250m-v1/` with all checkpoints (5K → 151,990, ~42 GB)
- `/workspace/eval-final.json` with the post-hoc eval result
- `/workspace/pilot.log` with the full pilot training log (with both pre-crash and post-resume sections)

If the user runs Stage 1.5, the pod will continue to be used. If they ship-as-is, the pod can be torn down (everything important is on R2 + GitHub).

### 13.9 One thing the next session should NOT do

**Do not re-trigger the pilot training run on the same `--checkpoint-root` without `--reset-data-position-on-resume`.** The pilot already exhausted the corpus at step 151,990. Resuming without resetting will immediately fire `training_complete` (zero new steps consumed; iterator empty).

If you want to continue the pilot's TRAINING (vs decay-only), you'd need a multi-epoch corpus reader (todo #4 in §13.7) or a fresh corpus.

### 13.10 Summary for the next session

- **Pilot done.** val_loss 2.878 / val_ppl 17.77.
- **Stage 1.5 ready to launch** but not started. User decision pending.
- **Three bugs fixed**, code at `dd7b202` on `main`.
- **Multi-epoch corpus** is the big Stage 2-blocking finding.
- All artifacts safe on R2.

If you're resuming with no context, your first command is:

```bash
ssh <pod>
tmux ls   # see what's running
tail -50 /workspace/pilot.log   # see where the pilot ended
cat /workspace/eval-final.json   # see the val_loss number
git -C /workspace/llm-build log --oneline -5   # see recent commits
```

That gets you oriented in 2 minutes.

---

## 14. SECOND POST-PILOT UPDATE (2026-05-15 ~20:00 UTC)

Updates since §13 was written. **All compute work is now done; pod has been torn down.** All artifacts on R2.

### 14.1 Stage 1.5 decay pass — LAUNCHED + COMPLETED

User decided to run Stage 1.5 (per §13.4) for WSD-decay code-path validation + small val_loss improvement.

```
Launched:  2026-05-14T22:33 UTC
Completed: 2026-05-15T00:51 UTC
Wall time: ~2h 18m on the same 4×H200 SXM pod
Cost:      ~$32
Final step: 171,990 (resumed from step 151,990 + 20K decay steps)
W&B run:   pilot-250m-v1-decay-2026-05-14 (id: pxoungh9)
```

**No NaN-skips or hard_spikes during decay.** WSD schedule fired cleanly; LR walked 3e-4 → 3e-5 linearly over the 20K steps. Eval hook fired ~20 times (~every 1000 steps) — the int32 fix from `dd7b202` confirmed working in production.

### 14.2 Post-decay eval results

Ran `scripts/eval_checkpoint.py` against the final step-171990 checkpoint on the 4×H200 pod:

```
val_loss : 2.730350     ← from 2.878 pre-decay (−0.147 nat improvement)
val_ppl  : 15.34        ← from 17.77 pre-decay (−13.7% perplexity)
```

This is **the official pilot final number**. Lands in my honest projection range (2.65-2.75). Stage 1.5 paid off.

### 14.3 G6 reshard fix — committed and validated in production

The pilot's checkpoint was saved on 4×H200 (DP=4). Trying to load on 1×H100 for generation testing triggered the long-known G6 bug at `checkpoint.py:143` (documented in §4 of original handoff, deferred Stage 2 prep). **Hit early; fixed in 3 iterative commits**:

```
13d6126  G6 reshard fix v1 (had stale orbax 0.6 `shape` kwarg)
3be12de  drop unsupported `shape` kwarg (orbax 0.7 API drift)
ca1c40b  force sharding on ALL leaves (orbax saves scalars as 0-d arrays too)
```

**Final pattern**:
- `CheckpointManager.restore()` now accepts optional `sharding: jax.sharding.Sharding`
- When provided, builds `restore_args` via `jax.tree.map(template, lambda x: ArrayRestoreArgs(sharding=sharding))` — for **every leaf**, including Python-scalar leaves that orbax stores as 0-d arrays
- Backwards-compatible: `sharding=None` (default) preserves the old behavior for same-mesh resume during training
- Callers in `scripts/generate.py` and `scripts/eval_checkpoint.py` auto-detect the current device topology

**Unlocks**:
- Loading pilot checkpoint on any device count (1×H100, 4×B200, 8×H200, etc.)
- Stage 2 mid-run pod resize (Stage-2 prep item retired)
- Stage 3 cross-mesh resume

### 14.4 Generation test — model verified working

After the G6 fix, ran `scripts/generate.py` (new file, commit `bc1d2b1`) on 1×H100 against the final checkpoint. **10 default prompts, ~$1 of compute.** Results:

**What WORKED** (genuine signal of recipe success):
- **Domain awareness**: code prompts → indented Python; theorem → LaTeX math; Wikipedia-style → encyclopedia format; news → news cadence
- **Multilingual**: "नमस्ते" produced coherent Hindi news article (sangraha data successfully integrated)
- **Code structure**: `def fibonacci(n):` continued with `return n + 1`, `def test_reorder(self):` etc. — Python style correct (content wrong)
- **Academic style**: "Theorem (Pythagorean):" produced LaTeX-formatted theorem-definition structure with `$S$`, `$k$`, etc.
- **No garbage**: every output grammatical, fluent, no `<unk>` spam, no NaN crashes
- **Style transfer per prompt**: different domains produce different (correct-style) outputs

**Expected weaknesses** (at this scale, no SFT):
- ❌ Factual recall: "capital of France" doesn't produce "Paris" (small model can't store many facts)
- ❌ Math: "2+2 = one plus one" — math reasoning emerges at 1B+ scale
- ❌ Looping after ~30-50 tokens (small base model classic; mitigated by larger scale + repetition penalty + SFT)
- ❌ No instruction following — it's a BASE model

**Verdict**: pilot validates the recipe. Multilingual + code + academic style all working. Stage 2's 1B at 10-30B tokens will fix most of the factual / reasoning weaknesses. SFT (post-release) addresses chat/instruction.

### 14.5 New scripts committed this round

| Commit | File | What |
|---|---|---|
| `bc1d2b1` | `scripts/generate.py` | Autoregressive text generation. 80 tokens / prompt default. Top-p + temperature OR greedy. Default 10 smoke-test prompts covering knowledge, code, Hindi, math notation. JIT-compiled forward over ctx-length padded buffer. |
| `13d6126` + `3be12de` + `ca1c40b` | `src/myllm/training/checkpoint.py` (final state at `ca1c40b`) | G6 cross-mesh restore fix. See §14.3. |

### 14.6 R2 state (verified before pod tear-down)

```
s3://llm-data/                                            Total: 161.30 GB
├── tokenizer/myllm-spm-unigram-131k-v2.json              4.79 MB
├── decontamination/                                       75 MB (8-gram + 13-gram)
├── corpus_v1_pilot/sources/                              20.08 GB (13 sources)
├── corpus_v1_pilot/train/                                20.09 GB (composed, 10 shards)
├── checkpoints/pilot-250m-v1/
│   ├── step-000005000 → step-000150000                   (31 checkpoints @ 2.65GB each)
│   ├── step-000151990/                                   ← Stage 1 FINAL (val_loss 2.878)
│   └── step-000151990/eval-final.json                    ← post-hoc eval result
└── checkpoints/pilot-250m-v1-decay/
    ├── step-000152000 → step-000170000                   (10 decay checkpoints)
    ├── step-000171990/                                   ← STAGE 1.5 FINAL (val_loss 2.730) ← THE BEST ONE
    └── step-000171990/eval-final-decay.json              ← post-decay eval result
```

The pilot final = `s3://llm-data/checkpoints/pilot-250m-v1-decay/step-000171990/`. **This is the artifact for the model card.**

### 14.7 Pod torn down — confirmed CPU-only work from here

- 4×H200 SXM pod: TERMINATED
- 1×H100 inference pod: TERMINATED
- Stage 1 / Stage 1.5 / generation testing all DONE
- $0/hr compute cost from here until next GPU need

**No state lost.** Everything reproducible from R2 + GitHub (`main` at commit `ca1c40b`).

### 14.8 Recent commits (since §13 was written)

| Hash | Subject |
|---|---|
| `ca1c40b` | **checkpoint: sharding required on ALL leaves, not just shape+dtype ones** |
| `3be12de` | checkpoint: drop unsupported `shape` kwarg from ArrayRestoreArgs |
| `13d6126` | G6 reshard fix: cross-mesh checkpoint restore via explicit sharding |
| `bc1d2b1` | scripts/generate.py — autoregressive generation from saved checkpoint |
| `d3ac216` | SESSION_HANDOFF_2026-05-14: append §13 post-pilot update |

Plus the original Stage 1.5 commits (`dd7b202`, `70b9009`, `9f442f7`) from §13.6.

### 14.9 NEXT-STEPS PLAN (the canonical roadmap, post-pilot)

**All work below can proceed without compute until Phase 3.** Pod can stay torn-down.

#### Phase 1 — Stage 2 enablement (engineering, no GPU, ~25-35 hr)

In rough priority order:

| Task | File(s) | Effort | Why it's a Stage 2 blocker |
|---|---|---|---|
| **Multi-epoch corpus reader** | `src/myllm/data/packed_corpus.py` | 4-6 hr | Stage 2 wants 10-30B tokens; pilot corpus is 5B. Single-epoch exhausts before target. |
| **P0-1: per-source val loss** | `src/myllm/training/eval_hook.py` + `scripts/run_pretrain.py` | 10-14 hr | Reviewer flag; diagnostic visibility during long runs |
| **P0-2: `--production` flag + fail-closed packed-corpus check** | `scripts/run_pretrain.py` | 2-3 hr | Reviewer flag; safety guard |
| **P0-3: packed-resume safety** | `src/myllm/data/packed_corpus.py::peek_data_position_from_checkpoint` | 2-3 hr | Reviewer flag; silent-restart bug |
| **Forward-only eval_step (FSDP-safe)** | `src/myllm/training/train_step.py` + `eval_hook.py` | 4-6 hr | Stage 2 needs FSDP; current eval breaks under FSDP via `donate_argnums` |
| **Cross-topology restore test** | `tests/test_checkpoint_reshard.py` (new) | 2 hr | Catch G6-class regressions before next deployment |

G6 reshard itself: ALREADY DONE (this session's `ca1c40b`).

#### Phase 2 — Writeup (no compute, ~10-15 hr)

| Task | Effort |
|---|---|
| Update `docs/governance/model_card_v1.md` with real numbers + sample generations + limitations | 3-4 hr |
| Update `docs/PROJECT_OVERVIEW.md` §2 scoreboard to "Stage 1 ✓ COMPLETE" | 1 hr |
| Public-facing "what we built" doc / blog post draft | 4-6 hr |
| Final reviewer packet for Stage 2 go/no-go | 2-3 hr |

#### Phase 3 — Release scorecard (small GPU run, ~$50)

| Task | Effort | Compute |
|---|---|---|
| Implement `predict_fn` in `src/myllm/eval/release_scorecard.py` (load checkpoint + greedy decode + benchmark scoring) | 4-6 hr | None |
| Run real benchmarks (MMLU-Pro, HellaSwag, ARC-Easy/Challenge, GSM8K, HumanEval-plus, IFEval, Belebele) against final pilot checkpoint | runtime | ~$50 on 1×H100 for full sweep |
| Post-process scorecard → JSON + Markdown → drop into model card | 1 hr | None |

#### Phase 4 — Stage 2 launch (after Phase 1 lands)

| Task | Effort | Compute |
|---|---|---|
| Stage 2 dry-run + canary on small pod | 4-8 hr | ~$20 |
| Final reviewer pass on Stage 2 readiness | external | none |
| Stage 2: 1B rehearsal @ 10-30B tokens | 3-5 days wall | ~$700-2000 on 4×B200 or 8×H200 |
| Stage 2 post-mortem + Stage 3 go/no-go | 1-2 days | none |

#### Phase 5+ — Stage 3 base run (much later)

Separate planning round after Stage 2 lands. Items:

- Compose v2 600B-token corpus build (~5 days, ~$50 R2)
- Teacher logit caching for DeepSeek-V4-Pro + Olmo-3-32B (~7 days, ~$500-1000 GPU)
- Real-text teacher audit re-run (~1 day, small GPU)
- Stage 3 launch: 1B @ 600B tokens with distillation (~30 days, ~$13K)

#### Background — Rust migration (parallel, separate person)

- v0.2.1 plan locked at `docs/stage3_rust_migration_plan.md`
- Implementer on `cs2hvh/llm-build-rust` fork
- Independent of Stages 1/2/3 timing

### 14.10 Recommended order for next session(s)

```
Week 1:  Phase 1 engineering — multi-epoch reader + P0-1/2/3 + forward-only eval_step + tests
Week 2:  Phase 2 writeup + Phase 3 scorecard ($50 GPU)
Week 3:  Stage 2 dry-run + launch (~$700-2000, 3-5 days)
Week 4+: Stage 3 prep
```

All Week 1 work is CPU-only. ~25-35 hr of engineering. Can do in a couple of focused workdays.

### 14.11 What the NEXT SESSION should know

- **Pilot is done.** Don't restart anything. The model is on R2.
- **Use commit `ca1c40b` (or later) on `main`** as the base. G6 reshard is in place.
- **First engineering target** = multi-epoch corpus reader. It blocks Stage 2.
- **No GPU needed until Phase 3**. Engineer on a cheap CPU box or local laptop.
- **Stage 2 decision gate** lands AFTER Phase 1 + 2 + 3 are done. Don't jump ahead.
- **DON'T try to restart pilot training**. Pilot is COMPLETE. There's no more useful training to do at 250M.

### 14.12 Summary (for the next session, in 4 lines)

```
1. Pilot DONE. val_loss 2.730 / val_ppl 15.34. Final ckpt on R2.
2. G6 reshard FIXED (cross-mesh restore works for any device count).
3. Generation VERIFIED on 1×H100. Model produces coherent text, multilingual ✓.
4. NEXT: Phase 1 = multi-epoch corpus reader + reviewer P0s, no GPU needed.
```

---

## 15. THIRD POST-PILOT UPDATE (2026-05-15, evening — Phase 1 done)

### 15.0 TL;DR

Phase 1 (the no-GPU engineering queue from §14.9) is **complete**, in 5 commits on `main`. Suite went from 642 → 674 tests (+32 new). All P0 reviewer asks shipped. The repo is now ready for Phase 3 (release-scorecard wiring + benchmark run, ~$50 of GPU) or Phase 4 (Stage 2 launch).

### 15.1 Commits, one-line each

| # | Commit | What |
|---|---|---|
| 1.1 | `be7574c` | Multi-epoch corpus reader — `iter_packed_pairs(epochs=N)`. Stage 2 unblocker (8 tests) |
| 1.3+1.4 | `082fa20` | `--production` flag + strict packed-resume safety (P0-2 + P0-3, 5 tests) |
| 1.5 | `107a551` | Forward-only `make_eval_step` — FSDP-safe, no donation, exposes per-token NLL (7 tests) |
| 1.6 | `97c59c1` | `tests/test_checkpoint_reshard.py` — pins all 3 G6 regression modes (6 tests) |
| 1.2 | `fbe9c72` | Per-source val loss (P0-1) via per-token NLL bucketed by DocSpan `source_id` (14 tests) |

### 15.2 What each delivers

- **Multi-epoch reader** (`src/myllm/data/packed_corpus.py:iter_packed_pairs`): `epochs=1` keeps legacy single-pass; `epochs=N` wraps `sid % total` for N cycles; `epochs=None` is unlimited. `data_position` stays monotonically increasing across epochs so resume is bitwise-exact. Stage 2 (1B at 10-30B tokens on a 5B-token corpus) needs `epochs=6+`.
- **`--production` flag** (`scripts/run_pretrain.py`): currently gates one fail-closed behavior — `peek_data_position_from_checkpoint(strict=True)`. Refuses to resume when the manifest is missing `data_position` (would otherwise silently re-feed already-trained data). More guards land here as Stage 2/3 prep items come up.
- **Forward-only `eval_step`** (`src/myllm/training/eval_step.py`): runs model+CE forward; no grads, no opt update, no donation. Compiles under the same `in_shardings` as `train_step` when `state_shardings=` is supplied, so live FSDP-sharded training state is reusable. Enabled `--eval-every` to work under `--fsdp` (was previously skipped with a warning). Also surfaces `nll_per_token` + `weight_per_token` for the per-source bucketer below.
- **G6 regression tests** (`tests/test_checkpoint_reshard.py`): pins the three failure modes from the 2026-05-14 fix iteration — (a) `shape=` kwarg gone from Orbax 0.7, (b) scalar leaves need `ArrayRestoreArgs(sharding=…)` not bare `RestoreArgs()`, (c) every leaf gets a sharding, no exceptions. Cross-mesh restore will fail loudly in CI if any of those regresses.
- **Per-source val loss** (`src/myllm/training/eval_hook.py`): `build_per_source_held_out(reader, n_sequences, micro_batch_size)` returns batches + per-label-position source-id arrays + a `source_name -> int` vocab. `make_per_source_validation_loss_eval_from_eval_step` consumes those plus the forward-only eval_step (with `return_per_token_nll=True`) to report aggregate AND per-source `val_loss` / `val_ppl` / `val_n_tokens`. Wired via `--per-source-val-loss` CLI flag; requires `--packed-corpus-root` (synthetic / on-the-fly data has no DocSpan provenance).

### 15.3 Things to know when launching the next training run

- **Stage 2 launch must use `--corpus-epochs 6` (or more)** to avoid the same single-pass exhaustion that ended the Stage 1 pilot at step 152K.
- **Always pass `--production`** for real training runs. Smoke + dev can leave it off.
- **For the model card per-source PPL table**, add `--per-source-val-loss --eval-every 5000` (or 2000 for Stage 1.5-style decay passes). Output keys are `val_loss/<source>` and `val_ppl/<source>` — they show up in W&B as nested metrics under `val/`.
- **FSDP eval is now safe.** No more `eval_hook_skipped_fsdp` warning. The forward-only path is selected automatically when `--fsdp` is set.

### 15.4 What stays open

- **Phase 3** (release_scorecard.py predict_fn + benchmark run) — ~$50 of H100 time. Yields the concrete benchmark numbers (MMLU/MMLU-Pro/HumanEval/IFEval/MATH/MBPP-Plus/MGSM/MMLU-ProX/Belebele) that the eventual model card and Stage 2 decision gate need.
- **Phase 4** (Stage 2 — 1B rehearsal at 10-30B tokens) — ~$700–$2000.
- **Phase 5** (Stage 3 distillation prep + launch) — ~$13K. Teacher caching needs to land; v2 corpus needs to be composed at scale.
- **Polish**: bump R2 `max_pool_connections`, add a RoPE drift test, stream from S3 instead of local stage, fix the per-source corpus parsers that have one-off issues.
- **[USER ACTION]** Harshit needs to request HF access for `ai4bharat/MILU` before we can add it to the decontam index.

### 15.5 What the test suite looks like now

```
$ pytest -q
==== 674 passed, 1 skipped, … warnings in ~48s ====
```

Phase 1 added: 8 (1.1) + 5 (1.3+1.4) + 7 (1.5) + 6 (1.6) + 14 (1.2) = 40 net (some are extensions to existing test classes). New test files added in Phase 1:
- `tests/test_eval_step.py` (7 tests)
- `tests/test_per_source_eval.py` (14 tests)

### 15.6 Don't forget when you read this

- Phase 1 is DONE. Don't redo any of those items.
- For the per-source PPL numbers in the model card, you'll need to *re-eval the pilot checkpoint* with `--per-source-val-loss`. That's a 1-GPU 5-min job on the final checkpoint (`step-000171990` from R2).
- Auto-memory pointer in `MEMORY.md` still says `project_session_state_2026-05-14.md`. That's fine — this section 15 + the Phase 1 commits *together* are what a fresh session needs.

