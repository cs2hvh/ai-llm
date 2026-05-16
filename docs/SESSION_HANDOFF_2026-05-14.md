# Session Handoff (current — last refreshed 2026-05-16)

> Filename keeps the `2026-05-14` suffix for auto-memory continuity, but
> this doc is **the live handoff for the next session**. Older
> minute-by-minute pilot narrative lives in `pilots/250m_v1/TIMELINE.md`;
> verified architecture/state in `docs/PROJECT_OVERVIEW.md`; pilot
> artifacts inventory in `pilots/250m_v1/R2_PATHS.md`.

---

## 0. TL;DR

1. **Stage 1 pilot is DONE.** 250M, 4×H200 SXM (DP-replicated; NOT FSDP), $385 spent. Final val_loss 2.7303 / val_ppl 15.34. Final checkpoint on R2 at `s3://llm-data/checkpoints/pilot-250m-v1-decay/step-000171990/`.
2. **Phase 1 engineering done** (5 commits, +32 tests, 642 → 674 suite). All P0/P1 items from pre-pilot review packet shipped.
3. **Two reviewer rounds processed** (2026-05-15 + 2026-05-16). Reviewer is technically correct on every claim I verified. Net: 3 doc errors in my packet + 4 P0/P1 code findings to fix.
4. **Stage 2 launch is gated** on Round A (quick wins) → Round B (KD gate + watchdog + benchmark-FSDP + scorecard predict_fn) → Round C (~$230 of GPU for benchmarks + μP sweep + decision gate). ~4 days + $230 before Stage 2 commit.
5. **DO NOT** commit Stage 2 ($700-2K) before Round C decision gate passes.

---

## 1. Where things stand (2026-05-16)

### Pilot

| Item | Value |
|---|---|
| Sharding | **DP-replicated** on 4×H200 SXM (no `--fsdp` in launch command). FSDP itself is proven through gauntlets G1-G4 + L2 parity + L3 canaries, just not used by the pilot. |
| Stage 1 final | step 151,990, val_loss 2.8776, val_ppl 17.77, corpus exhausted at single-pass boundary |
| Stage 1.5 final | step 171,990, val_loss 2.7303, val_ppl 15.34. Δ = -0.147 nats from pure decay phase |
| Watchdog | Quiet — 288 NaN-skip events across 172K steps (1.9 / 1K). Zero hard rollbacks. `lr_recovery_multiplier` stayed at 1.0 |
| Generation smoke | 1×H100, top-p 0.9, T=0.8 — coherent English + working Hindi. Factual recall weak (expected at 250M). |
| W&B runs | `roydqofb` (pre-crash) + `u5xsxm0l` (post-resume) + `pxoungh9` (Stage 1.5 decay) |
| Per-source PPL | **TBD — bundled with Round C1 GPU session** |

### Phase 1 (engineering queue from §14.9 of the old handoff — all DONE)

| # | Commit | What |
|---|---|---|
| 1.1 | `be7574c` | Multi-epoch corpus reader (`iter_packed_pairs(epochs=N)`) — Stage 2 unblocker |
| 1.3+1.4 | `082fa20` | `--production` flag + strict packed-resume safety (P0-2 + P0-3) |
| 1.5 | `107a551` | Forward-only `make_eval_step` — FSDP-safe, no donation, exposes per-token NLL |
| 1.6 | `97c59c1` | G6 cross-mesh restore regression tests (pins all 3 failure modes) |
| 1.2 | `fbe9c72` | Per-source val loss via per-token NLL bucketed by DocSpan source_id |

### Reviewer-cycle deliverables

| Date | Artifact | Where |
|---|---|---|
| 2026-05-15 | Post-pilot reviewer packet | `docs/review/POST_PILOT_REVIEW_2026-05-15.md` |
| 2026-05-15 | Round-1 audit reply from reviewer | (received in conversation, not in repo) |
| 2026-05-16 | Round-2 audit reply from reviewer (deeper, mostly correct) | (received in conversation, not in repo) |
| 2026-05-15→16 | Verification cross-checks | (chat history; key findings consolidated in §2 below) |

---

## 2. The plan ahead — Round A → B → C → D

### Round A — quick wins (TODAY, ~2-3 hrs, $0)

Removes every cheap reviewer criticism + closes the lingering P1 script lint.

| # | Fix | Where | Effort |
|---|---|---|---|
| A1 | 3 doc errors in reviewer packet (pilot FSDP claim; `ffn_dim: 3072` not 2048; 1B is 2048/8192/~1.24B not 1536/4096) | `docs/review/POST_PILOT_REVIEW_2026-05-15.md` | 15 min |
| A2 | Strip stale "SHARDING (planned, FSDP)" comment | `docs/PROJECT_OVERVIEW.md:370` | 5 min |
| A3 | `eval_checkpoint.py` jax import order — move `import jax` to top | `scripts/eval_checkpoint.py` | 10 min |
| A4 | `render_governance_cards.py` template-name fix | `scripts/render_governance_cards.py:292-293` | 10 min |
| A5 | `canary_ladder.py` packed-L3 default when `--packed-corpus-root` set | `scripts/canary_ladder.py:141` | 15 min |
| A6 | Hard-pin Orbax/JAX/TensorStore + restore() smoke test | `pyproject.toml` + new test | 45 min |
| A7 | Clarify pilot was DP-replicated (not FSDP) in pilot archive | `pilots/250m_v1/README.md` | 10 min |

### Round B — Stage 2 gating P0s (~1 day, $0)

Block a credible Stage 2 launch.

| # | Fix | Where | Effort |
|---|---|---|---|
| B1 | **KD vocab-mismatch fail-closed gate** — refuse `--distill` when teacher cache tokenizer SHA ≠ student tokenizer SHA | `scripts/run_pretrain.py` distillation setup | 1 hr |
| B2 | **Watchdog `_recover_from_spike`** — pass template + sharding through restore; advance `data_position` by skipped tokens | `src/myllm/training/loop.py:395-460` | 2 hr |
| B3 | **`benchmark_throughput.py --fsdp`** — mirror `run_pretrain.py`'s sharded init exactly | `scripts/benchmark_throughput.py` | 2 hr |
| B4 | **Phase 3 (a): scorecard `predict_fn`** — lift checkpoint+template+sharding load from `generate.py` into shared `src/myllm/infer/predict.py`, wire to scorecard | new module + edit scorecard | 3 hr |

### Round C — Stage 2 prep (~$230, ~4 days)

GPU spend that de-risks the Stage 2 commit.

| # | Step | Cost | Wall |
|---|---|---|---|
| C1 | Phase 3 (b): benchmark pilot checkpoint on MMLU-Pro / HumanEval+ / MBPP+ / IFEval / GSM8K. Also `--per-source-val-loss` to backfill packet TBD table | ~$30 | 1 day |
| C2 | Patched throughput bench on 1× and 8× H200 SXM at 1B shape × seq=4096/8192 (4 combos) | ~$50 | 1 day |
| C3 | μP/LR sanity sweep at 1B shape — 3 × 2K-step runs × peak_LR × 0.5/1.0/1.5 on 8×H200 SXM | ~$150 | 1 day |
| C4 | Decision gate: throughput in booking range + LR sweep clean → proceed | — | 30 min |
| C5 | **Stage 2 rehearsal**: 1B at 10-30B tokens on 8×H200, `--corpus-epochs 6 --production --per-source-val-loss --fsdp --use-chunked-ce` | $700-2K | 3-5 days |

### Round D — Stage 3 prep (parallel with Stage 2, no rush)

| # | Item | Effort |
|---|---|---|
| D1 | Chunked distillation (student CE+logZ through chunked path; teacher KL on top-K support only) | 1-2 days |
| D2 | Real-text teacher top-K mass audit (previous run was synthetic IDs — invalid) | 0.5 day + GPU |
| D3 | Stratified per-source held-out (sample K per source from shards, not corpus head) | 4 hrs |
| D4 | pg19 replacement (pile-of-books or rebalance) | depends on source |
| D5 | Stack Exchange `question + chosen_response` (currently question-only) | 2 hr |
| D6 | WSM evaluation overlay — `merge_recent(5)` + eval merged checkpoint alongside final | 1 hr |
| D7 | Logical-axis FSDP sharding rules (replace shape-heuristic in `mesh.py`) | 2-3 days |

---

## 3. Critical gotchas (silent-corruption modes to remember)

These are the bugs the pilot exposed. Each is now fixed + tested, but the *patterns* are worth carrying forward.

1. **int32 cursors in JIT'd train state** — `data_position` defaulted to int32 under JAX tracing and wrapped at step ~65,500 (2^31 ≈ 2.15B tokens at mb=4, seq=8192). Fix: pop from state before JIT call. Watch for other Python-int fields that could grow unboundedly.
2. **Single-pass iterators that look infinite** — `iter_packed_pairs` stopped at `total_sequences`. Pilot stopped 32 steps shy of intended end. Fix: `epochs=N` parameter. **Stage 2 MUST launch with `--corpus-epochs 6+`**.
3. **Orbax API drift bites in minor versions** — `ArrayRestoreArgs` dropped `shape=` between 0.6 and 0.7; scalar leaves need `ArrayRestoreArgs(sharding=...)`, not bare `RestoreArgs()`. Fix: G6 path in `checkpoint.restore(sharding=...)`. **Always pin Orbax exactly + smoke-test restore kwargs after any version bump.**
4. **FSDP `donate_argnums=(0,)` invalidates state for any subsequent caller** — using `train_step_fn` for eval-during-training under FSDP would silently corrupt training. Fix: forward-only `make_eval_step` with no donation, declared in_shardings only.
5. **Pilot ran DP-replicated, not FSDP.** Do not let docs/prose drift on this. FSDP is proven separately by gauntlets G1-G4; the pilot is end-to-end pipeline validation, not the FSDP performance evidence.

---

## 4. Reviewer findings — what's verified + still open

**Verified from round-2 review (2026-05-16)**:

| Finding | Status | Round |
|---|---|---|
| KD vocab mismatch — `cache_teacher_logits.py:163` clamps token IDs into teacher vocab; `loss.py:279` gathers student logits at teacher indices. Invalid for cross-tokenizer teachers. | OPEN — Round B1 | R1 + R2 |
| Watchdog `_recover_from_spike` — no template/sharding on restore; skipped batches don't advance `data_position` | OPEN — Round B2 | R1 + R2 |
| `benchmark_throughput.py` doesn't measure sharded path (zero `shard`/`fsdp`/`device_put` calls in the file) | OPEN — Round B3 | R2 |
| `build_release_scorecard.py::_build_predict_fn` raises `NotImplementedError` | OPEN — Round B4 (intentional scaffolding, Phase 3 work) | R1 + R2 |
| `eval_checkpoint.py:248` uses `jax.devices()` before `import jax` at line 298 | OPEN — Round A3 | R1 + R2 |
| `render_governance_cards.py:292-293` references `_template.md` files that don't exist | OPEN — Round A4 | R1 + R2 |
| `canary_ladder.py:141` hardcodes synthetic L3 even when `--packed-corpus-root` set | OPEN — Round A5 | R1 + R2 |
| Per-source held-out is from corpus head, not stratified | OPEN — Round D3 (P2 polish) | R2 |
| Chunked CE doesn't apply during distillation (`train_step.py:156` condition) | OPEN — Round D1 (Stage 3 prep) | R2 |
| `mesh.py` shape-heuristic (largest divisible axis or replicate) | OPEN — Round D7 (Stage 3 prep, long-run win) | R2 |
| Stage 1.5 manifest `data_position=655,360,000` is decay-local, not cumulative | NOTED — semantic gotcha | R1 |

**Hallucinated / overstated by reviewer**:

- Round 1: val_loss 2.502 (actual 2.7303). The decay improvement is 0.148 nats, not 0.376 nats as round 1 framed.
- Round 1: "PROJECT_OVERVIEW.md fractured" — false; the doc is consistent post-Phase-2 refresh. Drift was isolated to my reviewer packet (3 number errors).
- Round 2: "Documentation source of truth is fractured" — same as above; my packet had the errors, not the broader docs.

---

## 5. Resume / verify

```bash
cd /root/llm-build
.venv/bin/python -m pytest -q
# expect: 674 passed, 1 skipped
```

If suite fails, check git status first — local main is at `d06fe57`, origin/main matches.

For pilot inspection on a 1× GPU pod:
```bash
mkdir -p /workspace/ckpt/pilot-250m-v1-decay
aws --endpoint-url "$S3_ENDPOINT_URL" s3 sync \
  s3://$S3_BUCKET/checkpoints/pilot-250m-v1-decay/step-000171990/ \
  /workspace/ckpt/pilot-250m-v1-decay/step-000171990/

KERAS_BACKEND=jax .venv/bin/python scripts/generate.py \
  --model-config configs/pilot_250m.yaml \
  --tokenizer-path artifacts/tokenizer_v1.json \
  --checkpoint-root /workspace/ckpt/pilot-250m-v1-decay \
  --checkpoint-step 171990 \
  --prompt "The capital of India is" --temperature 0.8 --top-p 0.9
```

---

## 6. Where things live

| Need | Open |
|---|---|
| Pilot results, R2 paths, frozen configs, COMMANDS | `pilots/250m_v1/` |
| Canonical project state | `docs/PROJECT_OVERVIEW.md` |
| Reviewer packets + responses | `docs/review/` |
| Stage 3 Rust migration plan | `docs/stage3_rust_migration_plan.md` |
| muP design + scaling rules | `docs/mup_design.md` |
| Teacher distillation strategy | `docs/teacher_distillation_strategy.md` |
| Pilot model config | `configs/pilot_250m.yaml` |
| Base 1B config | `configs/base_1b.yaml` |
| Stage 1.5 decay config | `configs/pilot_250m_decay.yaml` |
| Per-source val loss code | `src/myllm/training/eval_hook.py` + `eval_step.py` |
| G6 cross-mesh restore | `src/myllm/training/checkpoint.py` |
| FSDP sharding | `src/myllm/training/mesh.py` |
| Watchdog | `src/myllm/training/watchdog.py` + `loop.py:_recover_from_spike` |

---

## 7. Recent commits worth orienting against

```
d06fe57  post-pilot reviewer packet (2026-05-15)
f9399e7  Phase 2: docs refresh — Phase 1 done, pilot done
fbe9c72  Phase 1.2: per-source val loss (P0-1) via per-token NLL
107a551  Phase 1.5: forward-only eval_step (FSDP-safe, no donation)
97c59c1  Phase 1.6: G6 cross-mesh restore regression coverage
ca1c40b  checkpoint: sharding required on ALL leaves, not just shape+dtype ones
3be12de  checkpoint: drop unsupported `shape` kwarg from ArrayRestoreArgs
13d6126  G6 reshard fix: cross-mesh checkpoint restore via explicit sharding
bc1d2b1  scripts/generate.py — autoregressive generation from saved checkpoint
dd7b202  Stage 1.5: decay-only continuation pass scaffolding
70b9009  eval_checkpoint: standalone post-hoc val_loss/val_ppl
9f442f7  loop: fix int32 overflow of data_position at ~2.1B tokens
082fa20  Phase 1.3 + 1.4: --production + strict resume safety (P0-3)
be7574c  Phase 1.1: multi-epoch corpus reader (Stage 2 blocker)
a6fde1a  pilots/250m_v1/: archival folder for the Stage 1 pilot
```

---

## 8. Contact

- **Lead**: harshit.hv@samatva.com (solo)
- **Auto-memory pointer**: `/root/.claude/projects/-root/memory/MEMORY.md`
- **R2 bucket**: `s3://llm-data/` (Cloudflare R2, endpoint in `.env`)

---

## 9. What the next session should NOT redo

- Pilot training. It's done. Don't restart.
- FSDP gauntlet. G1-G4 are PASS on 2× H200 SXM, regression-pinned.
- Phase 1.1 multi-epoch reader, 1.2 per-source val loss, 1.3+1.4 --production+strict resume, 1.5 forward-only eval_step, 1.6 G6 regression tests. All shipped.
- The reviewer's 12-question §8 from `POST_PILOT_REVIEW_2026-05-15.md`. The round-2 audit answered most of them.
- The reviewer-packet doc errors. Round A1 will fix them.
