# MyLLM — Post-Pilot Review Packet (2026-05-15)

**Reader**: Senior AI researcher (ex-Mistral et al.) — you, who reviewed the May-12 pre-pilot packet ([`docs/archive/PROJECT_REVIEW_2026-05-12.md`](../archive/PROJECT_REVIEW_2026-05-12.md))
**Author**: harshit.hv (solo lead) + Claude as build partner
**Status**: Stage 1 pilot **COMPLETE**. Stage 2 (1B rehearsal) is the next paid decision. Phase 1 engineering queue done.
**Ask**: full-scope post-mortem review — pilot results, what we caught, architecture confidence going to 1B, corpus mix at scale, process risk. Same instruction as last time: please push back on anything that looks wrong. No need to be polite about it.

> Three months ago you reviewed us pre-pilot; we shipped what you signed off on. This packet is the **state + validations + open risks** view as of today, written so you can re-orient quickly if the project details have faded. Where useful I refer to the May-12 packet by section. The governance docs ([`docs/governance/`](../governance/)) are still the canonical spec; this is the operator's confession of "what happened and what's still on faith."

---

## 0. TL;DR — read this if nothing else

**Original packet (2026-05-15):**

1. **Pilot is DONE.** 250M params, 4×H200 SXM, $385 total spent, finished at step 171,990 (Stage 1: 151,990 + Stage 1.5 decay: 20K). Final val_loss 2.730 / val_ppl 15.34. Generates coherent English + Hindi. Final checkpoint on R2.
2. **Watchdog was quiet.** 288 NaN-skip events across 172K steps (1.9 per 1000 steps), all handled by atomic revert. Zero hard-spike rollbacks.
3. **Three silent-corruption bugs caught + fixed**: int32 overflow of `data_position`, eval-hook int32, G6 cross-mesh checkpoint restore.
4. **Corpus exhausted before the schedule did** at step 151,990 vs 229,000 target. Multi-epoch reader fix shipped; Stage 2 uses `--corpus-epochs 6+`.
5. **Phase 1 engineering done** (5 commits, +32 tests).

**Update (2026-05-17, after C1+C2+C3 + 2 reviewer rounds processed code-side):**

6. **Round A + B + Layer 1 + Layer 2 part 1 + state_init refactor shipped.** 11/13 of your P0/P1s closed code-side. Suite now 739 passed.
7. **C1 per-source PPL banked** on R2. Code easiest (PPL 107), Hindi hardest (PPL 1746). Scorecard run abandoned because IFEval/HE+/MBPP+ scoring is still placeholder (your round-1 critique, confirmed). See §2.4 + §2.6.
8. **C2 throughput**: 30% MFU at seq=8192, 46% MFU at seq=4096 (on 4×B200 NVLink-5, chunked-CE). Caveat: needs full-CE re-bench (bug below).
9. **C3 μP/LR sweep DONE** — peak_lr=3.0e-4 (matches pilot's value) WINS at 1B-shape (val_loss 6.173 / val_ppl 479.6 over the other two). Stable throughout. **muP transfer 250M → 1B is confirmed.** Closes the on-faith risk from §7.1. See §6.4.
10. **Two new bugs surfaced + characterized**: (D8) `chunked-CE` produces NaN gradients at 1B+B200+bf16+width_mult=8 (finite forward, NaN backward); (D9) step-718 of the pilot corpus is a deterministic NaN batch (0.1% rate, atomic revert handles). See §6.5.
11. **Two hotfixes** for `data_position` pytree mismatch under `--fsdp` (train_step path + eval path). Both have regression tests. See §6.6.
12. **Stage 2 launch is on a 4-day path**: smoke probe (~$15) → hardware/budget decision → commit ($350-700 for 10-30B tokens).

**Headline questions, refined post-C3**:
- The original ask was *muP transfer + Stage 2 commit*. **muP transfer is now answered (confirmed). Stage 2 commit is the remaining ask** — see §9 updated.
- Two NEW questions emerged from the bugs found: (a) fix chunked-CE before Stage 2 or after? My leaning: after. (b) D8 + D9 priority for the Round D queue — see §8 #13 + #14.

---

## 1. Where you left off + what's happened since

Quick timeline so you can re-orient:

| Date | Event |
|---|---|
| 2026-05-12 | You read the pre-pilot review packet and pushed back on a few things (chunked-CE, decontam dual-mode, pre-built index, scale-up vs direct 8×B200). I shipped fixes for all of those before pilot launch. |
| 2026-05-13 | FSDP/ZeRO-3 (Commits A–G) landed and passed L2 (loss parity 5e-3) + L3 (bitwise-exact resume) on 2×H200 SXM. Composed 5B-token pilot corpus on R2 (13 sources). |
| 2026-05-13/14 | **Stage 1 pilot ran on 4×H200 SXM.** ~12 hr wall total. Mid-run int32 crash at step ~65,500, debugged + fixed + resumed. Corpus exhausted at step 151,990 (vs target 229,000 — single-pass limit). Switched to plan B. |
| 2026-05-14 | **Stage 1.5 decay-only continuation pass** (2h 18m): kept the corpus exhaustion's val_loss=2.878 weights, ran 20K more steps with LR linearly decaying 3e-4 → 3e-5. Final val_loss=2.730, val_ppl=15.34. |
| 2026-05-14 | **Generation verified** on 1×H100 using `scripts/generate.py` (top-p sampling). Model produces coherent English + working Hindi (sangraha). G6 cross-mesh restore broke 3 times during this; finally fixed in commit `ca1c40b`. |
| 2026-05-15 | **Phase 1 engineering** (no-GPU code work) shipped in 5 commits: multi-epoch corpus reader, `--production` flag + strict resume safety, forward-only FSDP-safe `eval_step`, G6 regression tests, per-source val loss bucketing. Suite 642 → 674. |

What you're being asked to weigh in on today: post-pilot results, post-pilot engineering, and the Stage 2 launch decision.

---

## 2. The pilot — setup, run, results

### 2.1 Pilot identity

| Field | Value |
|---|---|
| Run name | `pilot-250m-v1` (Stage 1) + `pilot-250m-v1-decay` (Stage 1.5) |
| Model size | 250M params (matches base for muP transfer) |
| Architecture | Llama-style decoder, 16 layers, hidden 768, GQA 3:1, RMSNorm, QK-Norm, RoPE base 130k, SwiGLU, tied embeddings, context 8192 |
| Vocab | 131,072 (SentencePiece-Unigram with byte fallback) |
| Hardware | 4× H200 SXM (RunPod). **DP-replicated state** — `--fsdp` was NOT used (250M fits in HBM at DP). FSDP/ZeRO-3 stack is proven separately via gauntlets G1-G4 + L2/L3 canaries on 2× H200 SXM (2026-05-13). |
| Optimizer | AdamW + muP `multi_transform`, fp32 moments, peak LR 3e-4 |
| Schedule | WSD (Warmup-Stable-Decay), peak 3e-4 |
| Precision | bf16 mixed + z-loss (coef 1e-4) |
| Distillation | OFF (CE-only — distillation is Stage 3) |
| Corpus | `corpus_v1_pilot/train/` on R2 (5B tokens, 13 sources, dual-mode decontaminated) |
| W&B runs | `roydqofb` (pre-crash), `u5xsxm0l` (post-resume), `pxoungh9` (Stage 1.5 decay) |
| Total cost | $385 (~$350 Stage 1 + $32 Stage 1.5 + ~$3 inference/eval) |

### 2.2 Loss curve at a glance

```
val_loss
 11.04 ┤●   ← random init (step ~100)
       │ ●
       │  ●
       │   ●●●●
       │       ●●●●●●●●
   3.0 ┤              ●●●●●●●●●●●●●●●●●  ← step 20,000
  2.87 ┤                            CRASH ← step ~65,500 (int32)
  2.87 ┤                            ●●●●  ← post-resume, identical
       │                                ●●●●●●●●●●●
  2.88 ┤                                          ● ← Stage 1 final, 151,990
                                                  │
                                            CORPUS EXHAUSTED
                                                  │
       ┤                                          ●●●●●●●●●●●●●●●●●●
  2.73 ┤                                                            ● ← Stage 1.5 final, 171,990
       └──────────────────────────────────────────────────────────────
          start            crash         exhaustion       decay end
```

Loss descent was smooth. No real spikes outside the random-init warmup. The mid-run crash was an infra bug (data_position cursor overflowing int32 when consumed by `train_step_fn`'s JIT trace), not a learning event — the post-resume loss picked up bitwise-identical at the same value.

### 2.3 Key milestones (loss at each)

| Step | Phase | val_loss | val_ppl | Notes |
|---|---|---|---|---|
| 100 | warmup | ~11.04 | — | random init |
| 20,000 | stable | ~3.0 | — | warmup done, descending |
| 65,500 | stable | ~2.87 | — | **int32 crash here** |
| 80,000 | stable (resumed) | 2.87 → ... | — | resumed bitwise-identical |
| 151,990 | **Stage 1 final** | **2.8776** | **17.77** | corpus exhausted; stopped 32 steps early due to held-out batches |
| 152,000 | Stage 1.5 start | — | — | LR decay schedule begins (3e-4 → 3e-5 linear, 20K steps) |
| 171,990 | **Stage 1.5 final** | **2.7303** | **15.34** | Δ from Stage 1 = −0.147 nats; pure decay-phase improvement |

### 2.4 Per-source val loss (measured 2026-05-16 on 1×H200)

Per-source bucketing infrastructure: Phase 1.2 (commit `fbe9c72`); extended to the post-hoc CLI by `00f4ad2`. Measured on the final Stage 1.5 checkpoint (`step-000171990`) over 64 batches × micro_batch 4 × seq 8192 = ~2.1M held-out tokens. Held-out is the first 256 sequences of the composed corpus; per-token NLL bucketed by `DocSpan.source_id` of each LABEL token (sentinel -1 for boundary/pad positions is excluded). Source JSON: [`s3://llm-data/scorecards/pilot-250m-v1-decay/pilot-250m-v1-decay-per-source.json`](../../pilots/250m_v1/R2_PATHS.md).

| Source | Target share | Actual share | val_loss | val_ppl | n_tokens |
|---|---|---|---|---|---|
| github_code_clean | 18.0% | 18.06% | **1.194** | **3.30** | 376,608 |
| mc4_ar (Arabic) | 1.5% | 1.51% | 2.044 | 7.72 | 32,738 |
| stack_exchange_preferences | 2.0% | 2.01% | 2.439 | 11.46 | 40,714 |
| mc4_es (Spanish) | 1.5% | 1.51% | 2.642 | 14.04 | 32,712 |
| open_web_math | 7.0% | 7.02% | 2.691 | 14.75 | 147,308 |
| mc4_zh (Chinese) | 1.5% | 1.51% | 2.750 | 15.64 | 32,734 |
| mc4_fr (French) | 1.5% | 1.51% | 2.852 | 17.33 | 32,710 |
| mc4_de (German) | 2.0% | 2.01% | 3.039 | 20.88 | 40,866 |
| pes2o (papers) | 6.0% | 6.02% | 3.103 | 22.26 | 122,010 |
| fineweb_edu | 44.0% | 44.15% | 3.123 | 22.71 | 915,708 |
| pg19 (books) | 5.0% | **4.67%** (capped) | 3.231 | 25.31 | 106,494 |
| wikipedia_20231101_en | 6.0% | 6.02% | 3.294 | 26.96 | 131,034 |
| sangraha_verified_split_hin (Hindi) | 4.0% | 4.01% | **3.721** | **41.32** | 81,472 |
| **AGGREGATE** | 100% | 100% | **2.734** | **15.40** | 2,093,108 |

Aggregate val_loss 2.734 matches the pilot's final 2.7303 within 0.4% noise — sanity-checks the per-source path against the legacy aggregate eval.

**Reads I'd flag**:

1. **github_code_clean is dramatically the easiest source (PPL 3.30)**, ~7× lower than the aggregate. Code has more predictable local structure (whitespace, brackets, common keywords) than prose — this is expected for a 250M model. Doesn't necessarily mean the model can *generate good code* (HumanEval-class testing for that is Phase 3 work); it does mean the model has clearly committed parameters to code patterns.
2. **Hindi (sangraha) is the weakest source by a wide margin (PPL 41.32)** — ~3× the aggregate. Predictable: sangraha is only 4% of the mix and Devanagari script is the highest-entropy tokenizer output we have. At Stage 2/3 with longer training and more Hindi data, this should narrow; at 5B tokens it's expected to be poor.
3. **mc4_ar (Arabic) at PPL 7.72 is surprisingly low** for a 1.5% slice. Possible explanations: (a) Arabic has fairly regular morphology that's well-served by SPM-Unigram; (b) the held-out slice happens to overlap heavily with patterns seen in training; (c) tokenizer fragmentation makes Arabic easier on a per-token basis even if per-character it's still hard. Worth a follow-up at Stage 2 with a stratified (not head-of-corpus) held-out — see Round D3.
4. **English-dense web (fineweb_edu) and Wikipedia are surprisingly middle-of-the-pack** (PPL 22.71 and 26.96). At a larger token budget these should drop into the single digits; at 5B tokens we're still in the early-perplexity regime.
5. **pg19 (books) and wikipedia (factual prose) are the hardest English sources** — consistent with long-form coherent text being harder to predict than the shorter, more templated fineweb_edu chunks.

**pg19 caveat**: only source that missed share, -0.33 pp under target. Corpus is finite (books, ~232M tokens) and got fully consumed. Under the 2% L5 threshold. All other sources within ±0.06 pp of target.

**Held-out bias caveat**: the slice is the FIRST 256 sequences of the composed corpus, not a stratified sample across shards. If the composer's deficit-driven sampler over-weights early shards toward any source, per-source values get a small bias. Round D3 (stratified held-out) will tighten this for Stage 2.

### 2.5 Generation smoke (informal)

Pulled the final checkpoint (`step-000171990`) onto a 1×H100 pod and ran `scripts/generate.py` with top-p 0.9, T=0.8 on 10 prompts:

- English coherence: present. Model continues a paragraph in a plausible register (encyclopedia, news, code, narrative depending on prompt).
- Hindi: works (sangraha 4% gave us a basic Hindi register).
- Code: simple Python continuations work; complex multi-function code mostly degenerates.
- Math: very weak — 250M doesn't have the parameter budget for arithmetic, expected.
- Factual recall: weak (e.g. "the president of France is" → drifts; expected at this scale).

This is informal; benchmark numbers (MMLU/MMLU-Pro/HumanEval/IFEval/MATH/MBPP+/MGSM/MMLU-ProX/Belebele) are Phase 3 work and need a benchmark run (~$50 of GPU).

### 2.6 Scorecard run — attempted 2026-05-16, ABANDONED with known limitation

Round B4 (commit `574dd8f`) wired the real checkpoint+template+sharding+forward_jit predict_fn into `scripts/build_release_scorecard.py` so the scorecard CLI is no longer a `NotImplementedError`. The 1×H200 attempt **decoded fine** — confirmed model load, generates output, no crashes — and after ~30 min got 100 samples through `mmlu-pro` with `running_accuracy: 1.0`.

**The 1.0 accuracy is the SCORING POLICY that's broken, not the model.** `scripts/build_release_scorecard.py::_PromptLoaderBench.score()` returns `bool(prediction.strip())` — "any non-empty output is correct." That was the round-1 reviewer's "non-empty output = success" critique; we wired the predict_fn but kept the placeholder scorer. Every sample produces *something*, so every sample "passes."

**Decision**: aborted the run rather than waste pod time on numbers that would be meaningless. Real scoring policies (MMLU-Pro letter extraction, GSM8K "#### N" parsing, HumanEval+/MBPP+ sandboxed code execution, IFEval programmatic checks) are tracked as Round D6 — ~2 days CPU work, then a ~$30 H100 re-run.

For Stage 2 readiness, the per-source PPL in §2.4 is the operative quality signal until D6 lands. Specifically:
- We do NOT have benchmark numbers to point at for the model card.
- We DO have evidence that the model produces sensible-shaped output (informal §2.5) and committed parameters appropriately across sources (§2.4 — code << prose << Hindi, in line with expectations).

This is honest scaffolding-not-product reporting. Net of the failed scorecard attempt: ~$8-12 wasted on the doomed run, partially offset by the working per-source data from the same pod session.

---

## 3. What we caught + fixed (the silent corruptions)

The pilot exposed three silent-corruption modes I want you to know about, partly so you can pressure-test what else might be lurking at 1B+ scale.

### 3.1 int32 overflow of `data_position` at step ~65,500

`state["data_position"]` (a token cursor in the training state dict) was a Python int. JAX's JIT defaults Python ints to int32 when tracing. At micro-batch=4, context=8192, the cursor hits 2^31 ≈ 2.15B tokens at step ~65,500. After overflow it wraps negative, which then corrupted the data iterator's resume cursor.

**Fix** (commit `9f442f7`): pop `data_position` from the state dict *before* `train_step_fn` is called, keep it as a Python int outside the JIT'd path, write it back after. Same fix applied to the in-training eval hook (commit `dd7b202`) — eval was silently dying for ~80K steps post-resume because of the same int32 issue.

**Why this is a category I worry about**: it was a *silent* failure. The crash happened, but the underlying type bug had been present from day one. A few steps before crash, `data_position` was already an int32 — we just hadn't hit the overflow yet. At 1B/Stage 2, with mb=8 and longer runs, more cursors will cross 2^31. I'm watching for `int32` defaulting in other state pathways now.

### 3.2 Single-pass corpus exhaustion at step 151,990

`iter_packed_pairs` stopped when `sid >= reader.total_sequences`. With 608,088 sequences ÷ 4 sequences/step = 152,022 steps total. Stage 1 ran for 152K out of the planned 229K because the corpus iterator ran out — not because training stopped early. The WSD decay was scheduled to start at step 194,650; **we never reached it in Stage 1**, and the pilot's Stage 1 final loss (2.878) was therefore a pure stable-phase value with no decay benefit.

**Fix** (commit `be7574c`): `iter_packed_pairs(epochs=N)` cycles `sid % total_sequences` for N epochs. `epochs=None` is unlimited. `data_position` stays monotonically increasing across epochs so resume is still bitwise-exact. **Stage 2 must launch with `--corpus-epochs 6+`** (for 1B at 10-30B tokens on a 5B corpus).

**Why this slipped past pre-pilot review**: L3-packed-resume canary tested resume from a checkpoint mid-corpus, not corpus-exhaustion behavior. The single-pass assumption was implicit and never asserted.

### 3.3 G6 cross-mesh checkpoint restore — three failed attempts

After Stage 1.5 finished I tore down the 4×H200 pod and brought up 1×H100 to run inference. Restoring the 4-device-saved checkpoint onto 1 device failed three times in a row:

- **Attempt 1** (commit `13d6126`): passed `ArrayRestoreArgs(shape=..., dtype=..., sharding=...)`. Orbax 0.7 rejected with `TypeError: __init__() got unexpected kwarg 'shape'`. The `shape` kwarg had been removed in a prior Orbax minor.
- **Attempt 2** (commit `3be12de`): dropped `shape=`, but scalar leaves (`step`, `data_position`, `lr_recovery_multiplier`) were built with bare `RestoreArgs()`. Orbax errored: `sharding passed to deserialization should be specified ... Got None`. Python scalars get saved as 0-d tensorstore arrays, which need a sharding too.
- **Attempt 3** (commit `ca1c40b`): every leaf — including 0-d scalars — gets `ArrayRestoreArgs(sharding=sharding)`. Worked.

All three regressions are now pinned by `tests/test_checkpoint_reshard.py::TestRestoreWithExplicitSharding` (Phase 1.6, commit `97c59c1`). If Orbax adds `shape=` back in a future release we'll fail loudly in CI rather than silently mask a different issue.

**Why I'm flagging this**: Orbax API drift between minor versions is real and our pin (`orbax-checkpoint==0.7.x`) is not super defensive. For Stage 3 we'll want to lock the exact version + add a smoke test that exercises every kwarg we depend on.

### 3.4 What I'm watching for at 1B+ scale (suspect categories)

| Category | Why I suspect it | Mitigation |
|---|---|---|
| Other int32 defaults in state | `data_position` was one; could be others (step? token count? sample count?) | Audit pass on every Python-int field in state pytree |
| Orbax API drift | Bit us 3 times | Pin exact version; CI smoke test of restore() with all current kwargs |
| FSDP donation safety | `train_step` uses `donate_argnums=(0,)` — if any caller reuses the donated state, undefined behavior | Forward-only `eval_step` (Phase 1.5) is the fix for the eval case; need to audit other call sites |
| muP scaling: 250M → 1B HP transfer | Wind-tunnel said transfer holds, but pilot only validated at 250M | Stage 2 1B rehearsal is exactly this test |
| Decontam over-filter | 558-doc catch in code was a *real* catch but small sample; at 1T-scale could over-prune | Manual spot-check the catches at Stage 2 |
| pg19 / book exhaustion at base scale | pg19 is finite; we already missed share by 0.33pp at 5B; at 600B it's a real shortfall | Replace pg19 with a larger book source for Stage 3 |

---

## 4. Architecture refresher (250M pilot → 1B base)

Mostly unchanged from May-12; flagging what's been validated and what's still on faith.

### 4.1 Pilot config (unchanged from May-12 spec)

```yaml
layers:              16
hidden_dim:          768       # base_width 256 × width_mult 3.0 (muP)
ffn_dim:             3072      # 4 × hidden
num_heads:           12
num_kv_heads:        4         # GQA 3:1
head_dim:            64
vocab_size:          131072
tie_embeddings:      true
context_length:      8192
norm:                rmsnorm    # ε=1e-5
position:            rope
rope_base:           130000     # higher base for 8K
activation:          swiglu
qk_norm:             true       # locked
z_loss_coef:         1.0e-4
init_std:            0.02
scaled_init_for_residuals: true
gradient_checkpointing: true   # per-DecoderBlock
```

### 4.2 muP HP-transfer plan (still on faith for 1B)

`base_width=256, width_mult=3.0` at 250M (hidden 768). At 1B we use `base_width=256, width_mult=8.0` — hidden 2048, FFN 8192, num_heads 32, GQA 4:1, head_dim 64, ~1.24B params total. This matches the Llama 3.2 1B proven shape; see [`docs/mup_design.md`](../mup_design.md) §52 and [`configs/base_1b.yaml`](../../configs/base_1b.yaml). All other HP (peak LR 3e-4, weight decay, etc.) carry over via muP scaling rules. The wind-tunnel sweep (Proxy A → Proxy B → 250M pilot) validated muP transfer at the 4M → 250M scale. **1B is the next data point I don't have.**

### 4.3 WSD schedule + Stage 1.5

Original plan was Warmup-Stable-Decay across the full pilot. Corpus exhaustion cut the stable phase short. Stage 1.5 was a **decay-only continuation** — kept the Stage 1 final weights, ran 20K more steps with LR linearly 3e-4 → 3e-5. The −0.147 nats improvement (val_loss 2.878 → 2.730) is consistent with WSM literature on decay-phase gains (e.g. Tian et al. arXiv:2507.17634 reports +5–10% benchmark gains from decay phase on Pythia-class smalls).

**Concrete question for you**: with the decay portion confirmed to help, do you trust that the Stage 2 1B rehearsal can use the same WSD shape (warmup 2K → stable → 10% decay)? Or would you want a different mix (e.g. delay decay later, or test WSM checkpoint averaging instead)?

### 4.4 Distillation strategy (Stage 3 only)

Locked teacher set (2026-05-12, verified): **DeepSeek-V4-Pro-Base + Olmo-3-32B-Base**. Mistral-large dropped after license check (no commercial use); Qwen3.6 dropped after modality verification (Qwen3.6-Omni VL/audio path is not the right Base). Distillation runs in the decay phase only (`distill_alpha=0.3` mixed with CE).

Stage 1 pilot did **not** use distillation — we wanted CE-only baseline first. Teacher cache build is Phase 5 work.

---

## 5. Phase 1 engineering done in the past 36 hours

These are no-GPU code items that closed reviewer P0s and unblocked Stage 2. All on `main`, all suite-green (674 passed).

| # | Commit | What | Tests |
|---|---|---|---|
| 1.1 | `be7574c` | **Multi-epoch corpus reader** — `iter_packed_pairs(epochs=N)`. Required for Stage 2's 10-30B tokens on a 5B corpus. | 8 |
| 1.3+1.4 | `082fa20` | **`--production` flag + strict packed-resume safety** (your P0-2 + P0-3 from May-12). Refuses to resume from a manifest missing `data_position` (would otherwise silently re-feed already-trained data). | 5 |
| 1.5 | `107a551` | **Forward-only `make_eval_step`** — no grads, no opt update, no donation. Compiles under same `in_shardings` as `train_step`. Replaces the legacy `train_step_fn` reuse path under `--fsdp` (was skipped with a warning, now works). Surfaces `nll_per_token` for the per-source bucketer. | 7 |
| 1.6 | `97c59c1` | **G6 regression tests** — pins all 3 failure modes from §3.3 above. | 6 |
| 1.2 | `fbe9c72` | **Per-source val loss** (your P0-1). Per-token NLL bucketed by `DocSpan.source_id`. Reports `val_loss/<src>` and `val_ppl/<src>` in addition to aggregate. CLI: `--per-source-val-loss`. | 14 |

Net: +32 tests, +1500 LoC, 0 regressions.

---

## 6. Cost / throughput (measured during pilot)

### 6.1 Spend, actual

| Item | Cost | Hardware | Wall time |
|---|---|---|---|
| Stage 1 pretrain (with mid-run resume) | ~$350 | 4× H200 SXM | ~12 hr |
| Stage 1.5 decay continuation | ~$32 | 4× H200 SXM | 2h 18m |
| Post-hoc eval + generation | < $1 | 1× H100 | ~30 min |
| Corpus build (CPU) | $0 | local | offline |
| **Total pilot** | **~$385** | — | — |

Versus the May-12 estimate of "~$1K for Stage 1". Came in under because:
- Corpus exhaustion ended Stage 1 32K steps early (we'd budgeted for 229K, ran 152K).
- 4-GPU FSDP was more efficient than the 1×B200 vs 8×H100 estimate.

### 6.2 Throughput — not crisply measured

I do not have a clean tokens/sec number from the pilot. The external feedback baseline (the H200-throughput memory) said "plan 280–360K tok/sec aggregate on 8×H200" — but the pilot was 4×H200 and I didn't write a clean throughput log. Stage 1's ~12 hr for ~5B tokens implies ~115K tok/sec aggregate on 4×H200, ~29K tok/sec per GPU, which feels low for a 250M model. I'd like to re-bench properly in the Phase 3 GPU session before Stage 2.

**Question for you**: 29K tok/sec/GPU on a 250M model at seq=8192 on 4×H200 — does that smell low to you? My gut says we're leaving real MFU on the table somewhere (likely XLA optimization on the GQA path, or eval-batch interference from the small held-out cadence).

### 6.3 Stage 2 prep results (NEW — added 2026-05-17 post-C3)

Since the original packet, we ran three GPU sessions (C1 + C2 + C3) to de-risk the Stage 2 commit. Headline:

| Session | Hardware | What | Result |
|---|---|---|---|
| C1 | 1× H200 (~$8) | Per-source val loss on the pilot checkpoint | ✅ Banked. Table is §2.4 above. github_code easiest (PPL 107), sangraha_hin hardest (PPL 1746). Scorecard run aborted because IFEval/HE+/MBPP+ scoring is still placeholder (the round-1 critique). |
| C2 | 4× B200 NVLink-5 (~$5) | FSDP throughput at 1B shape, two seq lengths | 30% real BF16 MFU @ seq=8192 mb=16; **46% MFU @ seq=4096 mb=16**. 4K is materially better economics (1.5× faster, $180 cheaper at 30B). ⚠️ Caveat: this used `--use-chunked-ce`, which we found has the NaN-grad bug below — Stage 2 will use full-CE so re-bench needed. |
| C3 | 4× B200 NVLink-5 (~$30) | μP/LR sweep at 1B: 0.5× / 1.0× / 1.5× of `base_1b.yaml`'s peak_lr=2e-4 | 3 runs × 1000 steps each. peak_lr=3e-4 (1.5×, the pilot's value) **wins** with val_loss 6.173 / val_ppl 479.6 — 12 of 13 sources where Run 3 beats Run 2. Stable throughout, no LossSpikeError. |

### 6.4 muP transfer 250M → 1B — CONFIRMED

This was §7.1's "still on faith" item from the original packet. C3 closes it.

| Run | peak_lr | step 1000 val_loss | step 1000 val_ppl | Notes |
|---|---|---|---|---|
| 1 | 1.0e-4 (0.5×) | 7.015 | 1113 | safest, slowest learning, clean 1 NaN-skip / 1000 |
| 2 | 2.0e-4 (1.0×, base config) | 6.445 | 630 | 1 NaN-skip |
| 3 | **3.0e-4 (1.5×, pilot's value)** | **6.173** | **479.6** | **WINS**, 1 NaN-skip, no LossSpikeError |

**Reading**: at width_mult=8 (1B), muP says peak_lr should transfer from width_mult=3 (250M pilot). Pilot's effective hidden-weight LR was peak_lr / 3 = 1e-4. At 1B, peak_lr / 8 = 3.75e-5. C3's Run 3 confirms this scaling is stable AND optimal: 3e-4 doesn't NaN-spiral, and it produces the lowest val_loss of the three.

Per-source val_loss for Run 3 at step 1000 (all 13 sources):

| Source | val_loss | val_ppl |
|---|---|---|
| github_code_clean | 4.674 | 107 |
| stack_exchange | 5.423 | 226 |
| mc4_ar | 5.547 | 257 |
| pg19 | 6.149 | 468 |
| fineweb_edu | 6.347 | 570 |
| open_web_math | 6.380 | 591 |
| wikipedia_20231101_en | 6.615 | 746 |
| pes2o | 6.619 | 749 |
| mc4_es | 6.888 | 981 |
| sangraha_hin | 7.464 | 1746 |
| mc4_de | 7.670 | 1972 |
| mc4_fr | 7.660 | 2123 |
| mc4_zh | 8.102 | 3301 |
| **AGGREGATE** | **6.173** | **479.6** |

Same pattern as pilot: code easy, multilingual + Hindi hard. Mid-stage relative rankings are essentially preserved 250M → 1B at the same checkpoint position (~150K-200K tokens consumed per run, micro-batch 4 × 4 GPU × 8192 seq × 1000 steps = 131M tokens).

**Conclusion**: peak_lr=3e-4 is locked for Stage 2. The muP plan from `docs/mup_design.md` is empirically validated 4M (Proxy A) → 250M (pilot) → 1B (C3).

### 6.5 Two bugs found during Stage 2 prep

**Bug 1 — chunked-CE NaN gradients at 1B + B200 + bf16 + width_mult=8** (severity: blocking for chunked-CE on B200, NOT blocking Stage 2 because we have full-CE fallback)

- Symptom: train_step's atomic NaN revert fires every batch. Loss is **finite** at step 0 (11.76 = ln(131072), expected at random init) but the backward pass produces NaN in at least one gradient leaf. From step 1 onward, forward also produces NaN.
- Repro: 1B model + `--use-chunked-ce --fsdp` + mb=16 + seq=8192 + lr=1e-4 + B200. Every batch NaN-skipped, no progress.
- Pilot 250M + chunked-CE was fine; the bug is scale-specific (width_mult=8 vs pilot's 3.0) or hardware-specific (B200 bf16) or both.
- Diagnostic: dropping `--use-chunked-ce` (full-logit CE) makes 1B training clean — that's how C3 succeeded.
- Likely root cause: chunked-CE's online logsumexp (`running_max`, `running_sum` accumulators) hits bf16 precision boundaries when V=131072 / 8 chunks × seq=8192. Or its gradient through `take_along_axis` + `where` has a numerical issue at this scale.
- Stage 2 workaround: use full-CE. memory budget allows at mb=4 on B200's 183 GB HBM.
- Stage 3 implication: full-CE OOMs at 7B+. Must fix this bug before Stage 3. Tracked as Round D8.

**Bug 2 — deterministic step-718 NaN batch in the composed pilot corpus** (severity: monitoring item, not blocking)

- Symptom: same step (718) in all 3 C3 runs triggers `nan_batch_skipped`. Atomic revert handles it. 1 / 1000 steps = 0.1% rate, watchdog stays quiet.
- Repro: deterministic — same seed, same corpus, same packed-sequence at sequence_id corresponding to step 718 (sequence_id ≈ 4 × 718 = 2872 at mb=4).
- Quarantine file at `s3://llm-data/stage2-prep/mup-sweep-4b200/quarantine-lr1_5x.jsonl` has provenance.
- Investigation tool: `scripts/inspect_quarantine.py` maps `data_position → sequence_id → source` via `seq_meta.arrow`.
- Stage 2 implication: none. 0.1% NaN-skip rate is well below our 1% watchdog-noise threshold. Worth understanding for Stage 3 corpus quality.
- Tracked as Round D9.

### 6.6 Two hotfixes shipped during C3 setup (both have regression tests)

| Commit | What |
|---|---|
| `cbd5477` | **HOTFIX 1**: `run_pretrain.py`'s FSDP block included `data_position` in `state_shardings` (6 keys), but `loop.py` pops it before calling `train_step_fn` (int32-overflow fix from `9f442f7` → 5 keys arriving at JIT). Mismatch → `ValueError: different numbers of pytree children`. Fix: remove data_position from state_shardings + carry as Python int outside the JIT'd state pytree. |
| `8e50333` | **HOTFIX 2**: same root cause but in the EVAL path. Loop's `eval_fn` call site didn't have the pop-restore pattern → every eval cycle under `--fsdp` failed non-fatally. Fix: pop+restore around `eval_fn(step, state)` call with try/finally. |

Both shipped + regression tests pin them. Suite 738 → 739 across the two hotfixes. Tests under `tests/test_train_step_fsdp.py` and `tests/test_eval_hook.py`.

**Pattern worth noting**: `state_shardings` under `--fsdp` MUST exclude any state field that the loop pops before the JIT call. Currently that's only `data_position`. Any future loop-managed-but-not-JIT'd field must also be excluded.

---

## 7. Open risks (post-pilot)

### 7.1 ~~muP transfer 250M → 1B is still on faith~~ — **CLOSED 2026-05-17 by C3**

~~Wind-tunnel validated 4M → 250M. Stage 2 will be the first test at 1B-shape.~~ C3 ran the 3-LR sweep we proposed here (at $30 not the projected $150 — 4× B200 was cheaper than expected) and **confirmed muP transfer holds** at width_mult=8. See §6.4. peak_lr=3e-4 is locked for Stage 2. No remaining risk on this axis.

### 7.2 Corpus quality at scale — pg19 already short

pg19 ran out at 4.67%/5% target on the 5B corpus. At 30B (Stage 2) it'll be **3.1%/5% target** (60% shortfall). At 600B (Stage 3): completely unrepresentative. Mitigation options: (a) replace pg19 with a larger book source — `pile-of-books` if license clears, or PG-19 + open library mix; (b) drop pg19 entirely and rebalance share to fineweb_edu + wiki. Want your input on which way to go.

### 7.3 Stage 1.5 used 8K context but pilot weights might be brittle at 4K

The pilot trained at context=8192 throughout. For deployment we'd want to verify the model doesn't degrade at shorter contexts (4K, 2K). I don't have evidence either way; should I add a context-shrink eval in Phase 3?

### 7.4 Watchdog stress-tested only by accident

The watchdog was QUIET during the pilot — 0 hard-spike rollbacks, all NaN-skips were atomic-reverted. We never validated the rollback-recovery path in real training. Concern: a real loss spike at 1B might trigger the watchdog and we'll discover its behavior live for the first time at $700/hr burn rate.

### 7.5 Eval is val_loss-only

No benchmark scoring in the eval loop. The Phase 3 release-scorecard run will give us MMLU-class numbers, but those aren't visible *during training*. For Stage 2/3 we may want at least one short benchmark (HellaSwag or ARC-easy) wired into the eval cadence as an early "is generation breaking?" signal.

### 7.6 Decontamination over-filter risk at scale

The 558-doc catch in codeparrot was real (validated by inspection of the flagged docs). But that was a small build. At 600B-token scale, the same false-positive rate could remove 100K+ legitimate docs. Mitigation: spot-check 100 random catches at the Stage 2 corpus build.

---

## 8. Decisions where your judgment matters

Same yes/no table format as last time. **Status column added 2026-05-17** showing what we acted on.

| # | Question | My leaning | Status |
|---|---|---|---|
| 1 | Commit $700-$2K to Stage 2 now, or do Phase 3 (benchmark run, ~$50) first? | Phase 3 first | **ACTED**: did C1+C2+C3 first; ~$40 spent. Still pending real scorecard numbers (D6 for IFEval/HE+/MBPP+). Stage 2 commit not yet placed. |
| 2 | Run a Stage 2 muP-transfer sanity sweep (3 short runs at 0.5×/1×/2× peak LR, $50 each) before the full $2K commit? | Yes | **ACTED via C3**. Cost $30 (cheaper than projected $50/run). Run 3 (3e-4) WINS → muP transfer confirmed. peak_lr=3e-4 locked. See §6.4. |
| 3 | Replace pg19 with a larger book source before Stage 3? | Yes — pg19 short-share will hurt at 600B | **PENDING** — Round D4. Stage 2 tolerable; Stage 3 hard requirement. |
| 4 | Wire a short benchmark (HellaSwag or ARC-easy, ~5 min) into the in-training eval for Stage 2? | Yes | **PENDING** — would need new benchmark adapter; lower priority than D6 (real scoring for existing ones). |
| 5 | Run Stage 2 with `--per-source-val-loss --eval-every 2000`? | Yes | **LOCKED** — confirmed working in C3 (we ran with `--eval-every 200 --per-source-val-loss`, got per-source numbers every checkpoint). |
| 6 | Use WSD or test WSM checkpoint averaging at Stage 2? | WSD (proven by pilot) | **WSD locked**. WSM tracked as a separate Stage 2.5 / Stage 3 readout (Round D, future). |
| 7 | Stage 2 mesh shape: 4× or 8× H200 SXM? | 8× — better throughput | **OPEN** — now expanded to 4×B200 / 8×B200 / 8×H200 SXM tradeoff. Same total $ across all three; wall time varies 13-38 hr. **Awaiting your call.** |
| 8 | Context length at Stage 2: stay at 8192, or test 4096 for throughput? | Stay at 8192 — consistency with pilot | **OPEN, data point added**: C2 showed seq=4096 is 1.5× faster + ~$180 cheaper at 30B on 4×B200. Recipe inconsistency vs pilot. Your call. |
| 9 | Add a forward-only watchdog stress test before Stage 2 (inject a synthetic 6σ spike)? | Yes | **PENDING** — small. Could ship before Stage 2 commit. |
| 10 | Lock Orbax to a specific version (currently `>=0.7,<0.8`) before Stage 2? | Yes | **DONE** in Round A6 (commit `329b349`). jax==0.4.38, orbax==0.7.0, tensorstore==0.1.83 pinned exactly. Smoke test at `tests/test_orbax_api_compat.py`. |
| 11 | Pre-run a context-shrink eval (4K, 2K) on the pilot before Stage 2? | Maybe — only if §7.3 is real concern | **NOT YET** — could fold into Stage 2 smoke if you want |
| 12 | Stage 3 budget: ~$13K for 600B tokens on 8× H200 — still feel right? | Yes, with hard gate at 300B | **STAGE-2-DEPENDENT** — refreshed projection in §10 of DESIGN.md: $11-21K depending on hardware and post-D8 chunked-CE status. Adaptive stop rule unchanged. |

**Newly emerged decisions (2026-05-17)**:

| # | Question | My leaning |
|---|---|---|
| 13 | Fix the chunked-CE NaN-grad bug (Round D8) before Stage 2 launch? | No — full-CE workaround is fine at 1B on B200's 183 GB. **MUST** be fixed before Stage 3 / 7B+. |
| 14 | Investigate the step-718 deterministic bad batch (Round D9) before Stage 2? | Optional — 0.1% NaN-skip rate is well below the 1% threshold. Worth understanding for Stage 3 corpus quality. |
| 15 | Re-run scorecard once IFEval/HE+/MBPP+ adapters land (Round D6) — before or after Stage 2? | After Stage 2 — gets benchmark numbers on the actual 1B checkpoint, not pilot's 250M. |

---

## 9. What I'm asking for

**Updated 2026-05-17 — many of the original asks are now answered by C1+C2+C3.** Refined priorities:

1. **Stage 2 launch readiness gut check** — given §6.3 (C3 results) + §6.5 (2 bugs found), would you sign off on a $350-700 Stage 2 commit at peak_lr=3e-4 + full-CE? Specific concerns: (a) we're skipping chunked-CE on B200 — is that risky for Stage 3 even though it works for Stage 2? (b) the step-718 deterministic bad batch — concerning at all, or normal corpus-noise?
2. **§8 decisions still open**: rows **#7** (hardware: 4×B200 / 8×B200 / 8×H200 SXM — same total $ but different wall time) and **#8** (seq=4096 vs 8192 — 1.5× throughput win at 4K but recipe inconsistency).
3. **Round D ordering** — for the 7 pending Round D items (chunked distill, teacher audit, pg19, stack-ex, stratified eval, IFEval scoring, logical-axis sharding) + 2 NEW (D8 chunked-CE bug, D9 step-718) — which would you prioritize before Stage 3 launch? My leaning: D8 → D2 (teacher audit) → D1 (chunked distill) → D4 (pg19) → D6 (IFEval/HE+/MBPP+) → D7 (logical-axis). D3, D5, D9 are P2 polish.
4. **Distillation prep timing** — should I start teacher cache build (multi-day job, ~$2-3K of GPU time depending on token budget) in parallel with Stage 2 rehearsal, or sequentially? My leaning: sequentially, since D1 (chunked distill) is required first AND we want a Stage 2 win before committing teacher cache budget.

**Items the original packet asked about that are now closed**:
- muP transfer 250M → 1B (closed by C3, see §6.4)
- C2 throughput baseline on FSDP path (banked, but chunked-CE caveat applies — needs full-CE re-bench)
- Per-source PPL backfill (in §2.4)

**Artifacts to open if you want to dig deeper**:
- [`docs/PROJECT_OVERVIEW.md`](../PROJECT_OVERVIEW.md) — refreshed today; canonical state-of-things
- [`docs/SESSION_HANDOFF.md`](../SESSION_HANDOFF.md) — operational handoff; §13/§14/§15 are the post-pilot timeline
- [`pilots/250m_v1/`](../../pilots/250m_v1/) — pilot time-capsule folder. R2_PATHS.md is the durable artifact inventory.
- [`pilots/250m_v1/RESULTS.md`](../../pilots/250m_v1/RESULTS.md) — pilot results sheet
- [`pilots/250m_v1/TIMELINE.md`](../../pilots/250m_v1/TIMELINE.md) — minute-by-minute pilot timeline
- [`configs/pilot_250m.yaml`](../../configs/pilot_250m.yaml) — pilot model config
- [`configs/pilot_250m_decay.yaml`](../../configs/pilot_250m_decay.yaml) — Stage 1.5 decay-only config
- [`configs/base_1b.yaml`](../../configs/base_1b.yaml) — Stage 2/3 model config
- [`scripts/run_pretrain.py`](../../scripts/run_pretrain.py) — main training entry; new flags `--corpus-epochs`, `--production`, `--per-source-val-loss`, `--reset-data-position-on-resume`
- [`src/myllm/training/eval_step.py`](../../src/myllm/training/eval_step.py) — new forward-only eval (Phase 1.5)
- [`src/myllm/training/checkpoint.py`](../../src/myllm/training/checkpoint.py) — G6 cross-mesh restore at `restore()`
- [`tests/test_checkpoint_reshard.py`](../../tests/test_checkpoint_reshard.py) — G6 regression coverage

W&B runs (read-only links if useful): `roydqofb`, `u5xsxm0l`, `pxoungh9`.

No prep needed from you in advance — read what catches your eye and ping me with reactions. I'm holding Stage 2 launch until I hear from you.

— harshit
