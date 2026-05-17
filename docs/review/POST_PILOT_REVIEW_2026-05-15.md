# MyLLM — Post-Pilot Review Packet (2026-05-15)

**Reader**: Senior AI researcher (ex-Mistral et al.) — you, who reviewed the May-12 pre-pilot packet ([`docs/archive/PROJECT_REVIEW_2026-05-12.md`](../archive/PROJECT_REVIEW_2026-05-12.md))
**Author**: harshit.hv (solo lead) + Claude as build partner
**Status**: Stage 1 pilot **COMPLETE**. Stage 2 (1B rehearsal) is the next paid decision. Phase 1 engineering queue done.
**Ask**: full-scope post-mortem review — pilot results, what we caught, architecture confidence going to 1B, corpus mix at scale, process risk. Same instruction as last time: please push back on anything that looks wrong. No need to be polite about it.

> Three months ago you reviewed us pre-pilot; we shipped what you signed off on. This packet is the **state + validations + open risks** view as of today, written so you can re-orient quickly if the project details have faded. Where useful I refer to the May-12 packet by section. The governance docs ([`docs/governance/`](../governance/)) are still the canonical spec; this is the operator's confession of "what happened and what's still on faith."

---

## 0. TL;DR — read this if nothing else

1. **Pilot is DONE.** 250M params, 4×H200 SXM, $385 total spent, finished at step 171,990 (Stage 1: 151,990 + Stage 1.5 decay: 20K). Final val_loss 2.730 / val_ppl 15.34. Generates coherent English + Hindi. Final checkpoint on R2.
2. **Watchdog was quiet.** 288 NaN-skip events across 172K steps (1.9 per 1000 steps), all handled by atomic revert. Zero hard-spike rollbacks. `lr_recovery_multiplier` stayed at 1.0.
3. **We caught three silent-corruption bugs the pilot would have failed slowly on:** (a) int32 overflow of `data_position` at ~65K steps, (b) eval hook silently dying after that fix due to same int32 issue, (c) G6 cross-mesh checkpoint restore broke under Orbax 0.7 API drift. All fixed + regression-tested.
4. **Corpus exhausted before the schedule did.** Single-pass reader hit `total_sequences=608,088` at step 151,990; we were targeting 229,000. Pivoted to a Stage 1.5 decay-only continuation on the same data. **The fix (multi-epoch reader, Phase 1.1) is shipped — Stage 2 will use `--corpus-epochs 6+`.**
5. **Phase 1 engineering done** (5 commits, +32 tests, suite 642 → 674). The repo is ready for Stage 2 launch once you sign off + we run the release-scorecard benchmark (~$50, Phase 3).

**Headline question I want you to weigh in on**: given §2 and §8 below, is the pilot evidence enough to commit $700–$2K to a 1B rehearsal at 10-30B tokens? Or are there gaps you'd want closed first (per-source perplexity numbers, ablations, more benchmarks)?

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

---

## 7. Open risks (post-pilot)

### 7.1 muP transfer 250M → 1B is still on faith

Wind-tunnel validated 4M → 250M. Stage 2 will be the first test at 1B-shape. If muP transfer doesn't hold, peak LR 3e-4 may be too high (or too low) and we waste $2K finding that out. Mitigation: run the first 1000 steps of Stage 2 with a low LR-multiplier sweep (3 quick runs of $50 each instead of a single $2K commit).

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

Same yes/no table format as last time. Each is something I have a leaning on but want your sanity check.

| # | Question | My leaning | Why I'd defer to you |
|---|---|---|---|
| 1 | **Commit $700-$2K to Stage 2 now, or do Phase 3 (benchmark run, ~$50) first?** | Phase 3 first — concrete benchmark numbers de-risk Stage 2 commit | $50 is cheap insurance against discovering the pilot model has a benchmark-floor problem after spending $2K |
| 2 | **Run a Stage 2 muP-transfer sanity sweep** (3 short runs at 0.5×/1×/2× peak LR, $50 each) before the full $2K commit? | Yes — first 1B-scale test | Cheap test of the on-faith assumption from §7.1 |
| 3 | **Replace pg19 with a larger book source** before Stage 3? | Yes — pg19 short-share will hurt at 600B | Stage 2 can tolerate it; Stage 3 should not |
| 4 | **Wire a short benchmark** (HellaSwag or ARC-easy, ~5 min) into the in-training eval for Stage 2? | Yes — early signal beats post-hoc | Don't want to be 50% through Stage 2 and discover generation is broken |
| 5 | **Run Stage 2 with `--per-source-val-loss --eval-every 2000`** to track which sources are improving? | Yes — Phase 1.2 infra is built | Cheap; lets us spot mc4_zh collapse early |
| 6 | **Use WSD or test WSM checkpoint averaging** at Stage 2? | WSD (proven by pilot) | WSM is in the codebase but unproven at our scale |
| 7 | **Stage 2 mesh shape**: 4× or 8× H200 SXM? | 8× — better throughput economics | 4× is what worked at pilot; but 8× lets us bench higher MFU |
| 8 | **Context length at Stage 2**: stay at 8192, or test 4096 for throughput? | Stay at 8192 — consistency with pilot | 4096 would be a 2× MFU win but changes the comparison |
| 9 | **Add a forward-only watchdog stress test** before Stage 2 (inject a synthetic 6σ spike)? | Yes — §7.4 risk | $5 of CPU time to validate the rollback path |
| 10 | **Lock Orbax to a specific version** (currently `>=0.7,<0.8`) before Stage 2? | Yes — see §3.3 | Three regressions in one day from API drift is too many |
| 11 | **Pre-run a context-shrink eval** (4K, 2K) on the pilot before Stage 2? | Maybe — only if you flag §7.3 as a real concern | $5 of GPU; cheap if useful |
| 12 | **Stage 3 budget**: ~$13K for 600B tokens on 8× H200 — still feel right to you given the pilot's economics? | Yes, with a hard gate at 300B | Adaptive stop rule from May-12 still applies |

---

## 9. What I'm asking for

In rough priority:

1. **Stage 2 launch readiness gut check** — given §2, §3, §6, §7, would you sign off on a $2K Stage 2 commit, or do you want Phase 3 + a muP sweep first?
2. **muP transfer confidence** — the on-faith 250M → 1B leap. Anything you'd want validated before spending $2K?
3. **Corpus mix at scale** — does the pilot mix look defensible for the 30B Stage 2 rehearsal? Should pg19 be replaced before Stage 2 or only before Stage 3?
4. **Process-risk pressure test** — given §3 and §3.4, what other silent-corruption modes would you suspect at 1B+? I'm specifically worried about FSDP donation safety in code paths I haven't audited yet.
5. **Decision pings on §8** — any of the 12 questions where one-word verdicts would unblock me.
6. **Distillation prep timing** — should I start teacher cache build (Stage 5 prep, multi-day job) in parallel with Stage 2, or sequentially after?

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
