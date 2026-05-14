# Session Handoff — 2026-05-14

**For the next session (you or another Claude) picking this up.** Read this first; it's the canonical resume pointer for whatever state the project is in.

**Last update:** 2026-05-14 13:00 UTC, mid Stage 1 pilot.
**Author:** Session that's running the live pilot. Replaces `docs/SESSION_STATE_2026-05-13.md`.

---

## 0. TL;DR — read this if nothing else

- **Project**: MyLLM — solo-lead enterprise effort to train a 1B-param decoder-only LLM from scratch. Llama-style, JAX/Keras 3, 131k vocab, ctx=8192.
- **Right now**: Stage 1 pilot (250M model) is **RUNNING on 4×H200 SXM** at step ~82K / 229K (~36% done, loss ~2.66). W&B run: [`roydqofb`](https://wandb.ai/harshit-hvpals-ahurasense/myllm/runs/roydqofb). ETA finish ~04:20 UTC 2026-05-15.
- **All artifacts on R2** at `s3://llm-data/`. Checkpoints mirrored every 5000 steps (16 so far at step-80K).
- **Critical bug fixed today**: int32 overflow of `data_position` (commit `9f442f7`). Pilot resumed cleanly from step-65000 checkpoint after the crash.
- **Hold pilot launch decisions** until friend's review of `docs/review/STATUS_2026-05-13.md` lands. Already running anyway — user chose to override the hold last night.

If you're picking this up cold, you only need:
1. Read this doc (~10 min)
2. Glance at `docs/PROJECT_OVERVIEW.md` for project canon (~10 min)
3. Glance at W&B run for current state (~1 min)

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
