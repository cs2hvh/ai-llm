# MyLLM Audit — Faithfulness Pass (2026-05-11, 6-week followup)

## Verdict

**Mostly on track, with two specific gaps that need attention before the pilot launches and a third that has grown from "P1" into "blocker for credibility."** The implementation work on R0/R1/R2/R3/R4/R5/R8 is unusually faithful to the dossier — almost no shortcuts were taken in the code itself. The risks have migrated: (a) the distillation path is wired through the train_step but **never gets activated by the training loop or by `run_pretrain.py`**, which is the actual production driver, so today the cache loader being a stub is the *second* hole; (b) **decontamination** is now the bottleneck on credibility — with MMLU-ProX/Belebele/MILU all reading from public HF datasets that overlap our web slot, the cost of shipping without a contamination report has gone up, not down; (c) pilot QK-norm being OFF while base + wind-tunnel are ON breaks muP's transfer assumption. None of these are hard to fix.

## Section A — Faithfulness audit of completed R-recommendations

### R0 (distillation) — 🟡 partial

Verified evidence:
- `src/myllm/training/loss.py` implements `cross_entropy_with_z_loss`, `kl_div_topk_loss`, `multi_teacher_kl_loss`, `distillation_mixed_loss`. The mixed loss correctly collapses to CE+z when teacher tensors are `None`, weights teachers by user-supplied or uniform vector, applies top-K KL with temperature, and respects `ignore_index`. Math is right (renormalised student over teacher's K positions; standard Hinton restricted-topK).
- `src/myllm/training/train_step.py` accepts `teacher_topk_logits` / `teacher_topk_indices` keys in the batch dict; falls back to CE when absent; `distill_alpha=1.0` default is safe.
- `src/myllm/data/teacher_cache.py` implements Arrow shard write/read, deterministic content-addressed naming, manifest read/write with coverage assertion, atomic tmp+rename, and bfloat16-as-uint16 round-tripping. Matches the locked binary spec in `docs/teacher_logit_cache_format.md`.
- `scripts/cache_teacher_logits.py` orchestrates producer with resume-from-manifest, top-K extraction via argpartition, synthetic-teacher mock for tests.

Holes (architecture-level):
1. **The training loop never activates distillation.** `src/myllm/training/loop.py` and `scripts/run_pretrain.py` (line 433) call `make_train_step(...)` with no `distill_alpha`, no `teacher_weights`, no `distill_temperature`. They also never inject teacher tensors into the batch dict. So today the production driver runs a CE-only loop *even when the cache exists*. The `decay_phase_distillation.yaml` config exists but is unread. Per its own docstring ("activation_fraction: 0.85 ... the training loop reads this YAML and switches the distill_alpha"), there's no consumer. This is the actual blocker, not the vLLM stub.
2. **Runtime cache reader doesn't exist.** `teacher_cache.py` is a shard-level writer/reader only; there is no `TeacherCacheReader.get_topk(positions: np.ndarray) -> (logits, indices)` mmap-based interface, although `teacher_logit_cache_format.md` §"Reader contract" promised exactly that. Without this you cannot stream teacher data into batches at training time.
3. vLLM producer is `NotImplementedError` — documented and accepted.

The architecture is still correct, but the integration surface area between cache → batch → train_step is missing one link. A loop-side switch like `if step >= int(activation_fraction * total_steps): batch = inject_teacher(batch, reader)` is ~200 LOC, but it has to land before Phase 4 decay.

### R1 (muP) — ✅ implementation, 🟡 recipe choice

Verified:
- `MupConfig` with `apply_*_output_mult` ablation knobs in `config.py`.
- All three output multipliers in `layers.py:299` (attn), `layers.py:358` (ffn), `transformer.py:88` (lm_head). All gated by `MupConfig` flags + multiply by `1/width_mult`.
- `label_variable_for_mup` and `build_optimizer` with `optax.multi_transform` correctly route `hidden` group through an `optax.scale(1/m)` after AdamW, leaving `embedding` and `norm` at base LR. Per-group LR scaling is mathematically what the EleutherAI minimal-muP recipe specifies.
- Wind tunnel config at `configs/wind_tunnel.yaml`: hidden_dim=384, base_width=256 → width_mult=1.5, 8 layers, 1B tokens/cell, 10-cell grid.
- `scripts/wind_tunnel_sweep.py` has plan/execute/collect modes, per-cell subprocess of run_pretrain with `--peak-lr-override` + `--init-std-override` (and run_pretrain consumes both via `model_copy`).

Recipe-choice question: **the EleutherAI "minimal muP" variant we picked is the right one for our scale band.** Init-scaling muP-pure adds three more moving parts for benefits EleutherAI's own ablations show are zero or negative below 7B. Yang et al.'s original derivation also shows the minimal variant is sufficient when output projections are explicitly scaled, which we do. **Hold.**

Risk: the sweep hasn't been executed, so we're shipping configs whose `peak_lr` is still 3e-4 / 2e-4 (Llama-band guesses) rather than wind-tunnel-derived.

### R2 (doc-masking + JAX FlashAttention) — ✅ faithful

Verified:
- `layers.py:255` routes through `jax.nn.dot_product_attention` when JAX backend (with cuDNN FlashAttention).
- `layers.py:246` builds `bool_mask = (same_segment AND causal)` from `segment_ids` when provided.
- `transformer.py:78-80` correctly suppresses the explicit causal mask when `segment_ids` is supplied.
- Manual fallback in `layers.py:268-295` preserves correctness on TF backend.

### R3 (QK-norm) — 🟡 inconsistency

- `layers.py:186-193` and `:228-230` apply per-head RMSNorm to Q and K post-RoPE (Llama-3 convention). ✅
- `base_1b.yaml:58`: `qk_norm: true` ✅
- `wind_tunnel.yaml:64`: `qk_norm: true` ✅
- `pilot_250m.yaml:49`: `qk_norm: false` 🔴

**This is a problem, and the comment at pilot_250m.yaml:49 ("decide based on pilot loss curve") is now stale.** If wind tunnel runs with QK-norm ON and pilot runs with it OFF, the wind tunnel's LR is no longer transferable to the pilot — the optimal LR shifts by ~10-30% with QK-norm changes, which is more than muP's transfer tolerance (~±10%). **Flip pilot to `qk_norm: true` before launching the sweep.**

### R4 (Nemotron-CC) — ✅ correct swap

Verified at `configs/data/pretrain_mix.yaml:24-32`:
- `tiiuae/falcon-refinedweb` is gone.
- `nvidia/Nemotron-CC` at `share: 0.135` (exactly Falcon's old share). ✅
- `config_name: HQ` to pull the high-quality subset only. ✅

### R5 (WSM) — ✅ faithful, 🟡 against Llama-3.2 baseline

Verified:
- `checkpoint.py:170-228` implements `merge_checkpoints(step_ids, output_step)` and `merge_recent(n, output_step)`.
- `_average_state_trees` averages **only** `trainable_variables` and `non_trainable_variables`. `step`, `opt_state`, `lr_recovery_multiplier` inherited from `states[-1]` (latest).
- `_is_merged` walks manifest extras to prevent meta-merges of already-merged outputs.

Equal-weight choice is right vs EMA; published-best at small-LM scale. Caveat: WSM paper recommends merging only the *stable-phase* (pre-decay) checkpoints; `merge_recent(n)` doesn't enforce that. Suggested: have the loop, when entering the decay phase, snapshot the last N stable checkpoints into a designated set.

### R6 (decontamination) — 🔴 **escalating to P0**

`src/myllm/data/decontamination.py` exists as a core n-gram-hash index, but:
- No benchmark prompt extractors exist (no glue from MMLU-ProX / Belebele / MILU benchmarks → the index).
- Not wired into the data pipeline.
- No CSV report emitted at gates.

**Priority escalation rationale:** Six weeks ago decontamination was "ship the contamination CSV at gates." But you've since wired MMLU-ProX, Belebele, MILU — every one of those is in the public web crawl. FineWeb-Edu and Nemotron-CC-HQ both contain mirrors of Belebele's FLORES passages and MMLU-style multiple-choice content. **Today you cannot trust any pilot or base eval number from those benchmarks because there is no decontamination filter.**

### R7 (long-context anneal) — not started

No `configs/longctx_anneal.yaml`. `base_1b.yaml:41` still says "extend later via YaRN" with no training step. **Sequence: AFTER Phase 2 pilot.**

### R8 (multilingual evals) — ✅ adequate for gates

`eval/runner.py`, `mmlu_prox.py`, `belebele.py`, `milu.py` all clean. Caveats: (1) MMLU-ProX HF ID is `li-lab/MMLU-ProX` — confirm at run-time; (2) MILU's answer-string-to-letter mapping will silently drop a percent or two of examples in non-Latin scripts — log the drop rate.

### R9 (EU AI Act doc) — not written

**Decision was correct.** Defer to Phase 12.

### R10 (teacher API legal posture) — ✅ done

`docs/teacher_distillation_strategy.md` is the source of truth.

## Section B — Still-pending items + priority calls

| Item | Old P | New P | Reason |
|---|---|---|---|
| R0 loop activation + cache reader | — | **P0** | Without these, distillation cost is incurred (cache = $15-25K) but model never benefits |
| R6 decontamination wiring | P1 | **P0** | Eval credibility blocker now that evals are wired in |
| Pilot `qk_norm: true` | — | **P0 (trivial)** | One-line config; muP transfer correctness depends on it |
| R5 stable-phase-only merge guardrail | — | P2 | Nice-to-have |
| R7 long-context anneal config | P1 | P1 (hold) | After pilot |
| R9 EU AI Act doc | P2 | P2 (hold) | After base |
| Wind-tunnel execution | — | P0 (compute) | Needs pod time |

## Section C — Newly-emerged issues spotted in the code

1. **No `BASE_PHASE` ↔ `DECAY_PHASE` switch in the training loop.** `run_pretrain.py` builds one `make_train_step` once and never swaps `distill_alpha`. Even if R0 cache reader landed tomorrow, you'd need either two train_step closures or a state-carried alpha parameter.
2. **`tokenizer.yaml` Hindi share inconsistency.** Tokenizer training uses `share: 0.20` for Sangraha-hin; pretrain mix uses `share: 0.04`. Intentional but undocumented. Add a comment.
3. **Wind-tunnel `qk_norm: true` + `mup.base_width: 256` creates a subtle issue.** QK-norm interacts with attention's softmax temperature in a width-dependent way. The minimal muP recipe assumes a fixed attention scale; with QK-norm, the effective scale is approximately width-invariant — so muP transfer still works, but only because of a happy accident.
4. **`merge_recent` does not validate that source checkpoints are stable-phase.**
5. **`scripts/run_pretrain.py:213` builds an `optax.join_schedules` WSD inline**, but `OptimizerConfig` has no `decay_fraction` or `warmup_steps` knobs — hardcoded. The lr_schedule section in `base_1b.yaml:71-76` is therefore *unused*. Either consume it, or remove the YAML stanza.
6. **`OptimizerConfig.peak_lr` default is 2.0e-4** but `run_pretrain.py:401-407` ignores both `base_1b.yaml:73`'s `peak_lr: 2.0e-4` and `pilot_250m.yaml:61`'s `peak_lr: 3.0e-4` unless `--peak-lr-override` is passed. **The pilot config's `3.0e-4` is silently overridden to `2.0e-4`. Bug.**

Items 5 and 6 are real bugs that will silently corrupt the pilot if launched today.

## Section D — SOTA developments since 2026-05-11

1. **Gemma 4 (released 2026-04-02)** — Apache-2.0 (dropped Gemma-3 restrictions). PLE trick (per-layer embeddings in flash). 128K context on small variants. **No reaction needed** — PLE is deployment-time, not training-time.
2. **Qwen 3.5 / 3.6** (Feb-Apr 2026) — Qwen 3.6-27B (already in our teacher set). No changes needed.
3. **DeepSeek-V4 / V4-Pro / V4-Flash** (Apr 2026) — confirms our teacher choice.
4. **No Llama 3.4/3.5/4. No SmolLM4. No Phi-5.**
5. **No new training-recipe paper as material as WSM or muP/muTransfer in the 6 weeks.**

**Net SOTA reaction**: nothing material.

## Section E — Top 3 things to do next

1. **Land R0 end-to-end integration (4-6 engineer-days):** (a) build `TeacherCacheReader.get_topk(positions)` on top of `teacher_cache.py` with mmap-backed shard lookup; (b) add a phase-switching wrapper in `loop.py` that, at `step >= activation_fraction * total_steps`, swaps the `train_step_fn` for a teacher-enabled closure; (c) wire `decay_phase_distillation.yaml` consumer in `run_pretrain.py`. **Without this, the existing R0 code is unfired hardware.**

2. **Wire decontamination into the data pipeline (3-5 engineer-days) and ship a contamination CSV at every Phase 2 gate.** Extract prompt strings from each `Benchmark` adapter, build a `DecontaminationFilter` that consults `DecontaminationIndex`, add it to `build_filter_chain`. Emit a CSV at gate evaluation time. **This is now the gating credibility item.**

3. **Pre-pilot config sanitization (1 engineer-day):** (a) flip `pilot_250m.yaml:49` to `qk_norm: true`; (b) make `run_pretrain.py` actually consume `model_cfg.lr_schedule.{peak_lr, warmup_steps, decay_fraction}` instead of using hardcoded defaults; (c) execute the wind-tunnel sweep (~$30-50, ~5 hours on 1× B200) and write the winning `(peak_lr, init_std)` back to `pilot_250m.yaml` and `base_1b.yaml`. Add `mup.base_width: 256` to both pilot and base configs **after the sweep validates muP works end-to-end**.

**Budget update:** the revised Phase 3 range ($80-115K incorporating the corrected ~$15-25K cache-generation cost) is correct. Once the wind-tunnel sweep runs, the $10-40K LR-rollback contingency can be retired.

---

*Audit produced by research subagent, 2026-05-11.*
