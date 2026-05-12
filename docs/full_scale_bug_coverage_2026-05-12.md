# Full-scale-only bug coverage map — 2026-05-12

The external reviewer's 2026-05-12 follow-up enumerated 10 bug classes that
only surface at full scale (real corpus, multi-day runs, real R2 uploads).
This doc maps each to its regression test so the next reviewer (and future-
us) can see the chain from bug → red test → fix.

| # | Bug class | Regression test | Fix location |
|---|---|---|---|
| 1 | data cursor reset on resume | `tests/test_data_cursor_resume.py::test_*` | Phase B re-audit Patch 2: `src/myllm/training/loop.py` advances `state["data_position"]` |
| 2 | optimizer state restore structure | `tests/test_orbax_multitransform_roundtrip.py::test_*` | Phase B B1: `src/myllm/training/checkpoint.py::restore(template=...)` rebuilds `MultiTransformState` |
| 3 | sequence length mismatch | `tests/test_wind_tunnel.py::TestSequenceLengthFromModelConfig::test_*` | Phase A P0-3: `scripts/run_pretrain.py` derives sequence_length from `model.context_length` |
| 4 | micro-batch config ignored | `tests/test_phase_b_reaudit_fixes.py::TestMicroBatchResolver::test_*` + `TestSweepMicroBatchPropagation::test_*` | Phase B re-audit Patch 3: `resolve_micro_batch` in `run_pretrain.py` + `--micro-batch-override` in `wind_tunnel_sweep.py` |
| 5 | NaN only after rare data batch | `tests/test_nan_skip_atomic.py::test_*` + `tests/test_quarantine.py::test_*` | Phase A P0-1: atomic NaN-skip via `jnp.where` in `src/myllm/training/train_step.py` + B6 `QuarantineWriter` |
| 6 | object-storage checkpoint partial write | `tests/test_full_scale_bug_coverage.py::TestCheckpointPartialWriteDetection::test_*` | manifest-last write order in `src/myllm/training/checkpoint.py::save` (existing) — pinned by new tests |
| 7 | teacher-cache offset mismatch on resume | `tests/test_full_scale_bug_coverage.py::TestTeacherCacheOffsetAlignment::test_*` | `SequentialCorpusPositions(start_position=...)` (existing) + position-count mismatch raises in `DecayPhaseActivation.maybe_inject` |
| 8 | per-source mixture drift | `tests/test_data.py::TestMixtureSampler::test_token_share_matches_target*` | Phase A P0-6: token-weighted deficit sampler in `src/myllm/data/mixture.py` |
| 9 | shape mismatch at decay-phase activation | `tests/test_full_scale_bug_coverage.py::TestDecayActivationShapeStability::test_*` | `DecayPhaseActivation.maybe_inject` returns same dict in stable phase; adds exactly `{teacher_topk_logits, teacher_topk_indices}` in decay phase, with documented shapes/dtypes |
| 10 | quarantine path unavailable | `tests/test_full_scale_bug_coverage.py::TestQuarantineGracefulDegradation::test_*` | New `_disabled` flag in `src/myllm/training/quarantine.py::QuarantineWriter.__init__` — degrades to no-op on `OSError` instead of killing the loop |

## Coverage at-a-glance

All 10 are now red-tested. 370 tests pass (4 skipped due to missing
optional `tensorflow` dependency in the keras-backend stack).

## What this does NOT cover (yet — explicit gaps)

The reviewer flagged some additional fault modes not in the "10 list" but
worth tracking for Phase 3 dress-rehearsal:

- **Object-storage rate limiting / 429 retries** at checkpoint mirror time
  → covered by `src/myllm/utils/storage.py` retries, but no chaos test yet
- **Multi-host partial checkpoint** (host 0 writes manifest but host 3 hangs)
  → single-host today; revisit at Phase 4 if we go multi-pod
- **Tokenizer.json byte-for-byte drift between pre-tokenize and train run**
  → B2 work — manifest will include tokenizer SHA256
- **Decontamination index revision drift** (index built at v1 of MMLU-Pro
  but training started against v2 split)
  → `decontamination_index.json` should carry dataset revisions; current
  format doesn't. Phase B follow-up.

## Process commitment (from reviewer's "Solo-lead failure pattern")

> Every bug found during reviews should become a permanent regression test or canary assertion.

This doc is the audit trail for that commitment. New bugs flagged in
future reviews get appended below, paired with the test path that pins them.

| Date | Bug | Source | Status |
|---|---|---|---|
| 2026-05-12 | Bugs #1-#10 above | External reviewer Q&A follow-up | All covered ✅ |
