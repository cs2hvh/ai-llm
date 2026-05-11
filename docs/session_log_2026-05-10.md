# Session Log — 2026-05-10
**Project**: MyLLM (sovereign-hedged 1B-parameter decoder-only LLM)
**Owner**: harshit.hv@samatva.com
**Status snapshot at end of session**: Phase 0 ✅, Phase 1 pipeline ✅, awaiting RAM upgrade to launch Phase 1 production training.

This doc is the **session handover** — if the chat closes/breaks, a future
Claude (or you, pasting this back as context) can resume from here.

---

## TL;DR (10 lines)

1. Phase 0 RunPod orchestration smoke **passed** — pod allocated, terminated, spend $0.0019.
2. Phase 1 tokenizer pipeline smoke **passed** — 6 sources stream, 131k Unigram trains, 8/8 round-trip tests OK including byte-fallback regressions.
3. Caught & fixed **4 production bugs**, the scariest being silent character-dropping in the tokenizer (Chinese/Arabic).
4. Wrote IndiaAI Mission compute-subsidy application brief (1-pager with `<TBD>` fields).
5. Wrote `scripts/bootstrap_pod.sh` to defer Docker registry push without losing reproducibility.
6. **87/87 tests still pass.**
7. User chose: **200K-doc/source production tokenizer training on this machine; Phase 2 pilot on B200** afterwards.
8. Flagged OOM risk at 200K/source on 32GB; user is **bumping/replacing this server to 64GB** before relaunch.
9. Full project tarball at `/root/myllm_backup_20260510-221218.tar.gz` (1.9MB, sha256 captured below).
10. **Awaiting user signal**: RAM bumped / new server ready → relaunch `train_tokenizer.py --samples-per-source 200000`.

---

## What was accomplished this session

### Phase 0 — RunPod orchestration smoke (DONE)

Command:
```bash
.venv/bin/python scripts/runpod_smoke.py --sku RTX_A4000
```

Result: pod `6gqbvjrxlogghk` (RTX A4000) — created → reached actual RUNNING (runtime allocated) in 33s → terminated cleanly. Total spend: **$0.0019**. Cost ledger at `artifacts/smoke_ledger.jsonl`.

### Phase 1 — tokenizer pipeline smoke (DONE)

Command:
```bash
.venv/bin/python scripts/train_tokenizer.py \
    --config configs/tokenizer.yaml \
    --output artifacts/tokenizer_smoke.json \
    --samples-per-source 500
```

Result:
- 6 sources streamed: fineweb-edu, codeparrot/github-code-clean, ai4bharat/sangraha (verified/hin), allenai/c4 (multilingual), open-web-math, wikimedia/wikipedia.
- 131,072 vocab, 256 byte-piece tokens + 24 special tokens.
- Tokenizer at `artifacts/tokenizer_smoke.json` (4.8MB), SHA-256: `3c1e9840af182a5c15a748d796adfd825b57ffd67c4ee0168639e131732ec3a8`.
- All 8 yaml validation round-trip tests pass.

### IndiaAI brief drafted
`docs/indiaai_compute_brief.md` — needs `<TBD>` fields filled (legal entity, team, affiliations).

### Bootstrap script written
`scripts/bootstrap_pod.sh` — installs deps on a stock Ubuntu/RunPod host. Substitutes for Docker registry push (which is deferred because no Docker Hub / GHCR creds yet).

### Test suite
`KERAS_BACKEND=jax .venv/bin/pytest -q` → **87 passed, 1 skipped (without backend), 23 warnings**.

---

## Bugs caught and fixed this session

| # | Bug | Symptom | Fix |
|---|---|---|---|
| 1 | `bid_per_gpu` no longer accepted by RunPod SDK 1.9 | `create_pod failed: got an unexpected keyword argument 'bid_per_gpu'` | Dropped from `to_runpod_payload()` in `src/myllm/runpod_orch/spec.py`. Field retained on spec for record-keeping. |
| 2 | `NVIDIA A10` retired from RunPod catalog; A100 SKU IDs also stale | `No GPU found with the specified ID` | Re-queried `runpod.get_gpus()`, verified all 12 SKU IDs in `GPUSku` enum, added B200/B300/L40S/A4000/H100_NVL. |
| 3 | Lifecycle declared pod RUNNING immediately (checked `desiredStatus` = the *goal*, not actual allocation) | Smoke "passed" in 1.5s with `ip=null` | `_wait_for_ready()` in `lifecycle.py` now requires `desiredStatus == "RUNNING"` **AND** `runtime is not None`. Real 33s allocation confirmed. |
| 4 | Tokenizer **silently dropped characters** absent from training corpus (Chinese 智, Arabic ص…) | "validation passed" log line, but `t.decode(t.encode("人工智能"))` returned `'人工能'` | 3-part fix: (a) inject 256 `<0xXX>` byte tokens as special tokens during training; (b) swap decoder from `Metaspace` to `Sequence[Replace, ByteFallback, Fuse, Strip]`; (c) post-process JSON to set `byte_fallback=true` **AND** demote byte pieces from `special:true` to `special:false` so `decode(skip_special_tokens=True)` (the default) doesn't filter them out before ByteFallback runs. |

**Why bug #4 was scary**: would have caused silent data corruption during 1T-token base pretrain. Caught at Phase 1 because hardened yaml validation (`docs/validation` round-trip section) now includes byte-fallback regression strings like `'人工智能是一门具有挑战性和创造性的学科。'` and `'الذكاء الاصطناعي يغير العالم بسرعة كبيرة.'`.

---

## Strategic decisions locked in earlier (unchanged this session)

- **Path B (English-primary)** + selective sovereign hedges. See `docs/playbook_alignment.md`.
- **Stack**: Keras 3 + JAX backend (`KERAS_BACKEND=jax`), Optax AdamW + WSD schedule, Orbax checkpointing, SentencePiece-Unigram tokenizer with NFKC + Metaspace + byte_fallback, GQA, RoPE base 500K, scaled init for residuals.
- **Storage**: Cloudflare R2 via boto3 (S3-compatible).
- **Tracker**: W&B.
- **Budget ceiling**: $15M; realistic projection ~$25-50K cloud-list, ~$15-30K with IndiaAI subsidy.
- **No Llama or Gemma derivatives anywhere** in training (license).

---

## Open state / pending tasks

### Immediate (awaiting user)
- [ ] **User to confirm RAM bump / new server is live** (target 64GB+ RAM).
- [ ] Once confirmed, relaunch `train_tokenizer.py --samples-per-source 200000 --output artifacts/tokenizer_v1.json` in background. ETA 1-3 hours wall-clock.

### Right after Phase 1 production tokenizer ships
- [ ] Round-trip + compression validation on the production tokenizer.
- [ ] Upload via `--upload` flag to R2 path `tokenizer/myllm-spm-unigram-131k-v2.json`.
- [ ] Compare per-language compression vs the smoke tokenizer; check `bytes_per_token_max_ratio_vs_cl100k: 1.6` gate.

### Phase 2 pilot (next major)
- [ ] Confirm user actually meant **B200** (they said "B100 not H100" but no B100 in RunPod catalog).
- [ ] Verify B200 availability via `runpod.get_gpus()` + pricing.
- [ ] Update `configs/pilot_250m.yaml` to use the production tokenizer.
- [ ] Pilot run: 1× B200, ~1B-5B tokens, validates the full data → model → checkpoint → R2 mirror pipeline on real GPU.

### Background / parallel tracks
- [ ] **Rotate the leaked credentials** (RunPod, HF, W&B, R2). They are in `.env`, in the chat history, and in `myllm_backup_20260510-221218.tar.gz`.
- [ ] Fill `<TBD>` fields in `docs/indiaai_compute_brief.md` (legal entity, team affiliations).
- [ ] Push Docker image to a registry (Docker Hub or GHCR) — deferred but needed before Phase 3 for cold-start speed.
- [ ] Add `KERAS_BACKEND=jax` to `pyproject.toml` `[tool.pytest.ini_options]` so the model test stops skipping by default. Polish, not blocking.

---

## Important paths

| Path | What |
|---|---|
| `/root/PLAN.md` | Master plan, ~600 lines, 14 phases, 4 hard gates. |
| `/root/llm-build/` | Project root. |
| `/root/llm-build/.env` | Live credentials (RunPod, HF, W&B, R2). **Mode 600. Leaked once via chat — rotate.** |
| `/root/llm-build/.venv/` | Python venv (1.4GB, not in backup, recreate via bootstrap_pod.sh). |
| `/root/llm-build/artifacts/tokenizer_smoke.json` | Working smoke tokenizer (8/8 round-trip). |
| `/root/llm-build/artifacts/smoke_ledger.jsonl` | RunPod spend ledger (~$0.002 total). |
| `/root/llm-build/configs/tokenizer.yaml` | Tokenizer config v3 — fixed sources after this session's debugging. |
| `/root/llm-build/configs/pilot_250m.yaml` | Pilot 250M config. |
| `/root/llm-build/configs/base_1b.yaml` | Base 1B config (matches Llama-3.2 1B). |
| `/root/llm-build/configs/data/pretrain_mix.yaml` | Pretrain data mix (separate from tokenizer corpus). |
| `/root/llm-build/scripts/train_tokenizer.py` | Tokenizer trainer with byte-fallback fix. |
| `/root/llm-build/scripts/runpod_smoke.py` | Phase 0 orch smoke. |
| `/root/llm-build/scripts/run_pretrain.py` | Pretrain launcher (uses WSD via Optax). |
| `/root/llm-build/scripts/bootstrap_pod.sh` | Stock-image bootstrap (deferred-Docker-push stop-gap). |
| `/root/llm-build/docs/playbook_alignment.md` | Path B + sovereign-hedges decision log. |
| `/root/llm-build/docs/architecture_review.md` | Architecture comparison vs Llama-3.2 1B / SmolLM2 / Qwen-2.5. |
| `/root/llm-build/docs/math_strategy.md` | Math handling across lifecycle. |
| `/root/llm-build/docs/safety_policy.md` | Refusal taxonomy v0.1. |
| `/root/llm-build/docs/indiaai_compute_brief.md` | IndiaAI Mission compute-subsidy application draft. |
| `/root/llm-build/docs/session_log_2026-05-10.md` | This file. |

---

## Backup tarball

| Field | Value |
|---|---|
| Path on this server | `/root/myllm_backup_20260510-221218.tar.gz` |
| Size | 1.9 MB |
| SHA-256 | `5c84ca639eb8bd9787e09f81b4c6eec7209a520e4317e6498f4b11fad9bcf451` |
| Contents | All of `llm-build/` (excluding `.venv`, `__pycache__`, `.cache`, intermediate smoke artifacts) + `PLAN.md`. Includes `.env` — **sensitive**. |
| Restore | `tar -xzf myllm_backup_*.tar.gz && cd llm-build && bash scripts/bootstrap_pod.sh && KERAS_BACKEND=jax .venv/bin/pytest -q` |

**Note**: This file (`docs/session_log_2026-05-10.md`) was written AFTER the backup tarball. Either regenerate the tarball before resize, or grab this file separately.

---

## How to resume

### If a future Claude session is reading this
1. Read this file end to end. Read `/root/PLAN.md` and `docs/playbook_alignment.md` for project context.
2. Read `docs/architecture_review.md` if you need to make architectural decisions.
3. Check the **Open state** section above — that's the queue.
4. **Don't** redo work already in the **What was accomplished** section.
5. The user prefers: enterprise-grade rigor; honest gap audits over false confidence; bug fixes diagnosed at root cause rather than papered over; concise responses; markdown links to files with line numbers.
6. The user has chosen Path B with sovereign hedges. Do not push them toward sovereign positioning unless they explicitly ask.

### If the user is pasting this into a new chat
Paste this entire file as your first message, prefixed with:
> *"This is a session-handover doc from my previous Claude session on the MyLLM project. Read it and resume from the 'Open state' section. Confirm you've read it before doing anything."*

---

## User-preference notes (to apply next session)

- **Treat as enterprise project**, not a learning project. Reproducibility, auditability, and root-cause fixes matter more than fast iteration.
- **No Llama or Gemma derivatives** in training pipeline (license).
- **English-primary** with Hindi + Spanish + Chinese + Arabic + French + German as secondary languages.
- **Credentials hygiene**: user pasted live credentials in chat once and committed to rotating "later"; the leaked ones are still in use until they rotate. Treat the `.env` as compromised.
- **GPU pick for Phase 2**: B200 (user said "B100 not H100" — no B100 in RunPod catalog, defaulting to B200).
- User uses VSCode + Claude Code extension. References to files should use markdown links: `[filename.py:42](path/to/file.py#L42)`.

---

*End of session log. Generated 2026-05-10 22:1X UTC by Claude (Opus 4.7).*
