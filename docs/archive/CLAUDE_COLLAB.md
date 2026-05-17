# Claude ↔ Claude Sync — collaboration brief

**Purpose**: User runs two Claude sessions in parallel. This doc is the
async handoff between them. Read it before touching code; update the
"WHO'S DOING WHAT" section when you start/finish a substantive task so
the other session knows what's safe to touch.

**Last updated**: 2026-05-13 ~18:25 UTC (Session A)
**Source of truth for project state**: [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) and [review/STATUS_2026-05-13.md](review/STATUS_2026-05-13.md)
**This doc**: just coordination + live state + what's safe to grab.

---

## Quick orientation (you've already read PROJECT_OVERVIEW.md, this is just the moment)

MyLLM — 1B-param decoder-only foundation model, solo lead `harshit.hv@samatva.com`, treated as enterprise. Today (2026-05-13) we:

- ✅ Validated FSDP on 2× H200 SXM (G1-G4 PASS, the correctness gates)
- ✅ Wired dual-mode decontam (8+13 gram) through the build
- ✅ Built + uploaded code source (codeparrot/github-code-clean, 180M tokens earlier today)
- ✅ Shipped eval hook (val_loss + perplexity, blended across sources)
- ✅ Wrote release scorecard scaffold
- ✅ **Pilot dry-run on synthetic data PASSED** (250M model builds, eval fires, ckpt saves)
- 🟢 **5B-token pilot corpus build RUNNING NOW** on this 128-core box

User has shared the project with a **friend reviewer**. Reviewer report (`reviee13.pdf`, audit hash 355f054) verdict:
- Stage 1 pilot: **Conditional Go** — launch when corpus is ready
- Stage 2/3: GO only after per-source val loss + cross-mesh ckpt portability + production fail-closed guards

**User decision (explicit)**: skip the reviewer's P0 items for now. Launch pilot once corpus finishes. P0 items land before Stage 2.

---

## WHO'S DOING WHAT (live coordination)

> **Edit this section as you grab/finish work. Format: claim + ETA.**

### Session A (terminal where the build was kicked off — the one this doc was written by)
- **HOLDING** `/workspace/corpus_pilot_build/`, `/tmp/parallel_build.log`, the running build process (PID 918051 orchestrator + ~5 children at ~18:25 UTC)
- **HOLDING** `configs/pilot_250m.yaml`, `configs/data/pretrain_mix_pilot.yaml`, `docs/pilot_corpus_rebuild_plan.md` (mid-iteration on the ctx=8192 decision)
- **MONITORING** the build, will run compose pass when it finishes
- **WAITING** on user for: GPU pod choice for Stage 1 pilot launch

### Session B (you, just joined)
- (claim work here)

### Don't both edit simultaneously
- `docs/PROJECT_OVERVIEW.md` — the canonical doc. Whoever updates it, claim it first.
- `docs/review/STATUS_2026-05-13.md` — friend-reviewer-facing. Same rule.
- `configs/data/pretrain_mix_pilot.yaml`, `configs/pilot_250m.yaml` — pilot-critical, will be loaded by training. Claim before editing.
- `scripts/run_pretrain.py` — central training entry point. Claim before editing.
- `src/myllm/training/loop.py` — central loop. Claim before editing.

### Safe to grab right now (no current holder)
- `docs/CLAUDE_COLLAB.md` (this file) — update freely; treat like a shared whiteboard
- Anywhere in `tests/` — add tests freely
- Reviewer P0/P1/P2 items (full list below)
- Polish items: RoPE drift fix, docs/config sync test, watchdog stress test
- New scripts that don't conflict with existing flows
- Documentation polish in `docs/` for files NOT listed above

---

## Active running processes you must NOT kill

| PID | What | Started | ETA |
|---|---|---|---|
| 918051 | `run_parallel_builds.py` orchestrator | 17:39 UTC | ~1 hr remaining |
| (children) | per-source `build_packed_corpus.py` (currently fineweb_edu, sangraha + bg uploaders) | 17:39 | bound by fineweb_edu @ 2.2B target |

If you need to kill it, use `kill -9 918051` then `pkill -9 -f build_packed_corpus`. Don't do this without user confirmation.

If you need CPU for something heavy: the build is at ~14% CPU (13 single-core processes). 115 cores idle. You can use them.

---

## The reviewer's checklist (DEFERRED per user, but track here)

Reviewer (friend) report at `reviee13.pdf`. All 4 P0 claims verified accurate via code inspection. User said skip them for now; land before Stage 2.

### [BEFORE STAGE 2] items
- **P0-1** Per-source val loss (`val_loss/<source>`). Eval hook today is blended-only. Need to enhance `src/myllm/training/eval_hook.py` + `scripts/run_pretrain.py` to bucket by source_id. Est 10-14 hr.
- **P0-2** `--production` flag + fail-closed packed-corpus check in `scripts/run_pretrain.py` (doesn't exist there; only in `build_packed_corpus.py`). Est 2-3 hr.
- **P0-3** Packed-resume fail-closed invariant. `peek_data_position_from_checkpoint` returns 0 for both "no ckpt" and "old ckpt without data_position" — silent restart-from-0 risk. Est 2-3 hr.
- **P1** Real GPU cross-mesh checkpoint portability pass (G6 from gauntlet). Has real orbax bug at `src/myllm/training/checkpoint.py:143` — `restore()` call lacks `RestoreArgs(sharding=...)`.

### [POLISH] (any time)
- RoPE doc drift: `PROJECT_OVERVIEW.md` says `RoPE base=130000`, `configs/base_1b.yaml` says `500000.0`. Fix the doc.
- `tests/test_docs_config_sync.py`: catch future drift before merge
- Branch protection / PR workflow (process; user-owned)
- CI gate for L3 packed resume + L2 parity canary
- Watchdog stress test against real orbax checkpoint

### [USER ACTION needed]
- **MILU dataset gated on HF**. User must request access at https://huggingface.co/datasets/ai4bharat/MILU before MILU can be added to decontam index / eval gate.

---

## Today's commits (chronological — what landed since session start)

| Hash | Subject |
|---|---|
| cc56daa | Decontam: wire DualModeDecontaminationIndex through corpus build |
| aaff2fb | mmlu-prox adapter: read column-shaped option_0..option_9 schema |
| (decontam index extend to 10 benchmarks, R2 re-upload — no commit) | |
| b1dd5dd | Teacher top-K mass audit + code-supplement yaml + 14 tests |
| b10b64b | build_packed_corpus: docstring no longer claims compose is TODO |
| eab98c9 | GPU pod bring-up + FSDP gauntlet + teacher audit runners |
| 258f97f | fsdp gauntlet: force micro_batch divisible by GPU_COUNT |
| f5c7a8a | fix gauntlet JSON aggregator + audit defaults |
| 6240bec | fix audit JSON aggregator |
| 6414c5e | audit: install accelerate + correct DeepSeek-V2-Lite repo id |
| 99cde30 | pin jax[cuda12]==0.4.35 (later bumped) |
| b71be2d | bump pin to jax[cuda12]==0.4.38 |
| 8ebbc87 | gauntlet G4/G6 fixes |
| 7cfddb6 | audit: fail fast if torch CUDA missing |
| d65ba4d | torch: pull from cu124 + cusparselt (later replaced) |
| 6cf299e | **canonical single-venv recipe: torch==2.7.1 + jax[cuda12]==0.4.38** |
| 190062f | G5 throughput: compute from step timestamps |
| 7002d50 | gauntlet: force fresh checkpoint root |
| 172d971 | docs: 2026-05-13 status snapshot + PROJECT_OVERVIEW updates |
| 355f054 | STATUS doc: lift review-ask to top |
| 518aa50 | **eval-during-training: --eval-every wires val loss + perplexity** |
| 03a8b3a | pilot corpus plan: 5B target, codeparrot-swap yaml |
| 7331377 | pilot: D2 — bump pilot ctx 4096→8192 |
| fb6a537 | release scorecard scaffold + 9 tests |

Reviewer saw up to 355f054. They did NOT see 518aa50 (eval hook), 7331377 (ctx=8192), fb6a537 (scorecard). Their critique still stands on per-source val loss though (our hook is blended, not source-aware).

---

## R2 paths you'll want to know

```
s3://llm-data/
├── tokenizer/myllm-spm-unigram-131k-v2.json       # production tokenizer
├── decontamination/
│   ├── decontamination_index_8gram.json           # 1.75M ngrams, 10 benchmarks
│   └── decontamination_index_13gram.json          # 1.74M ngrams, 10 benchmarks
├── corpus_v1/sources/                             # OLD pre-today builds at seq_len=8192 (WRONG; off-by-one P0 bug)
│   ├── fineweb_edu/                               # ~440M tokens, NOT USABLE for new training
│   ├── codeparrot-github-code-clean/              # 180M tokens (built today at seq_len=8193 — usable for ctx=8192)
│   └── (11 other sources at WRONG seq_len)
├── corpus_v1_pilot/sources/                       # NEW build in progress (this morning's plan)
│   ├── fineweb_edu/                               # building → 2.2B target
│   ├── github_code_clean/                         # DONE, 900M
│   ├── (rest of 13 sources)                       # mostly DONE; sangraha + fineweb still running
│   └── (TODO) train/                              # compose pass output, runs after build
└── corpus_v1_pilot/train/                         # not created yet
```

The OLD `corpus_v1/sources/*` (except codeparrot) are at `seq_len=8192` (wrong — should be ctx+1=8193). Don't use them for new training; we're rebuilding everything at 8193 in `corpus_v1_pilot/`.

---

## Environment + secrets that must be set on any GPU pod

```bash
# R2 / Cloudflare (the .env file in repo root has these on the dev box)
export AWS_ACCESS_KEY_ID='...'
export AWS_SECRET_ACCESS_KEY='...'
export S3_BUCKET='llm-data'
export S3_ENDPOINT_URL='https://<account>.r2.cloudflarestorage.com'
export AWS_DEFAULT_REGION=auto    # ← CRITICAL; without this, R2 uploads fail with "region 'nov' invalid"

# HuggingFace
export HF_TOKEN='hf_...'

# JAX cuDNN shadowing fix (RunPod images pre-set LD_LIBRARY_PATH to /usr/local/cuda/lib64)
unset LD_LIBRARY_PATH
```

---

## Canonical GPU pip stack (DON'T deviate without research)

```
torch==2.7.1
torchvision==0.22.1
jax[cuda12]==0.4.38
transformers>=4.46
accelerate>=1.0
```

That combo is the single-venv answer for driver 12.8+. Rationale: torch 2.7.1's strict `==` nvidia-* pins (CUDA 12.6 ABI) satisfy JAX 0.4.38's loose `>=` pins. Single `pip install`, no `--force-reinstall`, no `--no-deps`. Per [apple/axlearn](https://github.com/apple/axlearn/issues/858).

**Anti-patterns** (we hit ALL of these today, hours of debugging):
- ❌ `pip install --force-reinstall jax[cuda12]==X` → corrupts `nvidia/cuda_nvcc/` namespace
- ❌ `pip install --upgrade nvidia-cudnn-cu12` → pulls 9.22.x → JAX `CUDNN_STATUS_NOT_INITIALIZED`
- ❌ Torch from `whl/cu124` → 2.6.0 with cuDNN 9.1.0.70, fights everything else
- ❌ Default PyPI torch → CUDA 12.9+ build → driver 12.8 too old
- ❌ Forgetting `unset LD_LIBRARY_PATH` → system cuDNN shadows JAX's bundled one

---

## File ownership / who's expert on what (you can ask in code, this is just hint)

| Area | Primary file(s) | Last touched today (key commits) |
|---|---|---|
| FSDP / sharding | `src/myllm/training/mesh.py`, `optimizer.py`, `train_step.py`, `checkpoint.py` | pre-today (Commits A-G) + 8ebbc87 |
| Eval hook | `src/myllm/training/eval_hook.py`, `loop.py` | 518aa50 (today) |
| Release scorecard | `src/myllm/eval/release_scorecard.py`, `scripts/build_release_scorecard.py` | fb6a537 (today) |
| Decontam | `src/myllm/data/decontamination.py`, `scripts/build_decontamination_index.py` | cc56daa |
| Pilot corpus build | `configs/data/pretrain_mix_pilot.yaml`, `docs/pilot_corpus_rebuild_plan.md` | 7331377 (today) |
| Gauntlet | `scripts/run_fsdp_gauntlet.sh`, `scripts/pod_launch_gpu.sh`, `scripts/run_teacher_audit.sh` | eab98c9, multiple fixes |
| Compose | `src/myllm/data/compose.py`, `scripts/compose_mixed_corpus.py` | pre-today, verified today |
| Audit | `scripts/audit_teacher_topk_mass.py`, `scripts/cache_teacher_logits.py` | b1dd5dd, 7cfddb6, 6414c5e |

---

## Coordination conventions

### Commit messages
Style we've used today: subject line ≤ 70 chars, blank line, body in paragraphs (not bullets unless it's a list of distinct fixes). Body explains WHY + cites source observation (`pod 2026-05-13`, `reviewer P0-X`, etc.) when relevant. End with empty line, no signoff.

### When to ask user vs decide
- **Decide alone**: doc updates, test additions, code refactors that don't change behavior, bug fixes obviously aligned with their direction.
- **Ask first**: anything that costs them money (pod spin-up), anything destructive (`rm`, force-push), anything they were just deciding (don't override).
- **Verify before assuming**: anything the reviewer or a previous Claude said. User got burned today by reviewer claims that turned out true (verified them all) but also burned earlier in week by an unchecked claim.

### Memories already saved (auto-memory system at /root/.claude/projects/-root/memory/)
- User role + stance, enterprise-rigor preference, control-plane pattern, verify-before-locking
- External resources for MyLLM (R2, HF, .env state)
- Tokenizer trainer (native SentencePiece for production)
- Teacher strategy v2 (DeepSeek-V4-Pro-Base + Olmo-3-32B-Base — both hypothetical/future names)
- uint32 for 131k vocab
- H200 throughput baseline (280-360K tok/sec, not 520K)
- No --force-reinstall on nvidia extras
- Canonical GPU pin (torch 2.7.1 + jax 0.4.38)
- FSDP validated 2026-05-13

If you discover something durable, append to `MEMORY.md` + write the file. Don't duplicate.

### Don't blow up the pod
Anything that does `rm -rf` on /workspace or kills a running build → ask user first.

---

## Open questions / decisions pending user

1. **GPU pod for Stage 1 pilot**: user said "decide later when build finishes"
2. **Whether to land reviewer P0 items before pilot**: user said NO (skip, just launch)
3. **Stage 1 token target**: 10B (pilot stack-validation) vs 30-50B (default per pilot_250m.yaml). User hasn't picked.
4. **Pilot context length**: we set ctx=8192 (D2 decision — same as base v1 so corpus is reused). Hasn't been validated by user against the wind_tunnel-era ctx=4096 expectation.

If user comes to your session asking about these, refer them to this doc (you can read the answers above), not the other Claude.

---

## Stuff I'm watching right now (session A handoff dump)

```
PID 918051 — corpus build orchestrator
   ETA: ~1 hr remaining, bounded by fineweb_edu @ 2.2B target
   Output: /workspace/corpus_pilot_build/sources/<source-id>/ (deleted-after-upload)
   R2: s3://llm-data/corpus_v1_pilot/sources/
   Logs: /workspace/build_logs/<source>.log per-source
   Launcher log: /tmp/parallel_build.log
   To check: pgrep -f build_packed_corpus | wc -l   (should be > 0 until done)
   To check progress: grep build_one_source_done /workspace/build_logs/*.log | wc -l
```

When all 13 build_one_source_done events fire, the next step is the compose pass:

```bash
cd /root/llm-build && source .venv/bin/activate && set -a && source .env && set +a
# First sync the per-source corpora from R2 to local (compose reads from local)
# OR, since compose runs against local /workspace/corpus_pilot_build, just compose if files still there
python scripts/compose_mixed_corpus.py \
    --sources-root /workspace/corpus_pilot_build/sources \
    --output-dir /workspace/corpus_pilot_build/train \
    --pretrain-mix-config configs/data/pretrain_mix_pilot.yaml \
    --sequences-per-shard 65536 \
    --strict-sources \
    --corpus-name corpus_v1_pilot_train
# Then sync the compose output to R2:
aws --endpoint-url "$S3_ENDPOINT_URL" s3 sync \
    /workspace/corpus_pilot_build/train s3://$S3_BUCKET/corpus_v1_pilot/train/
```

If `--delete-local-after-upload` already ate the per-source data, the compose needs R2 source-pull first. Check `/workspace/corpus_pilot_build/sources/*/manifest.json` exists; if not, sync sources back from R2 before compose.

---

## How to brief the user when both sessions report

Pick a consistent identifier. I'll be **Session A**, you be **Session B** in user-facing messages, so they can map a status report to its source.

Format that works well for the user:
- Lead with the headline (what changed)
- Then 1-2 sentences of context
- Concrete next-action options (use AskUserQuestion if a real choice)

Avoid: redundant restates of what's in PROJECT_OVERVIEW; emojis; long preambles; "Great question!" openers.

---

## Final note

User treats this as enterprise (per memory). Prefer root-cause fixes + honest gap audits. Surface silent-corruption bugs. Don't be overly polite — point out problems directly. If you find an issue this doc doesn't cover, write it down here so future Claude (you, me, or another session) doesn't repeat it.
