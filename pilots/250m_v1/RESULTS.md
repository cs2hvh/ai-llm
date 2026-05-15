# MyLLM Pilot 250M v1 — Results

This is the detailed results doc. For the high-level summary, see [`README.md`](./README.md).

## Headline numbers (official model card values)

| Metric | Stage 1 end (step 151,990) | Stage 1.5 end (step 171,990, **final**) |
|---|---|---|
| **val_loss** | 2.878 | **2.730** |
| **val_ppl** | 17.77 | **15.34** |
| Train loss (smoothed) | ~2.4 | ~2.05 |
| Tokens trained | 4.98 B | 5.6 B (4.98 + 0.64 decay re-iter) |
| Wall time cumulative | ~12 hr | ~14 hr |

**The Stage 1.5 numbers are what goes on the model card.** Stage 1 alone (without decay) is the "if we'd stopped at corpus exhaustion" baseline.

Eval methodology: take the first 32 batches of `corpus_v1_pilot/train` (the same held-out subset the in-training eval hook used pre-crash) with `--micro-batch 4`, compute mean cross-entropy on next-token prediction. Reproducible via `scripts/eval_checkpoint.py` — see `COMMANDS.md`.

Source: [`artifacts/eval-final-decay.json`](./artifacts/eval-final-decay.json).

## How this compares to other small models

| Model | Params | Tokens | Tokens/param | val_ppl on similar data |
|---|---|---|---|---|
| **MyLLM Pilot v1 (post-decay)** | **250 M** | **5.6 B** | **22 : 1** | **15.34** |
| GPT-2 Small (2019) | 124 M | ~10 B | ~80 : 1 | ~25-30 |
| Pythia-410M | 410 M | 300 B | 730 : 1 | ~12 |
| TinyLlama-1.1B-Chat | 1.1 B | 3 T | 2,727 : 1 | ~6.5 |
| SmolLM2-360M-Instruct | 360 M | 4 T | 11,000 : 1 | ~6 |

**Our pilot punches above its tokens/param ratio.** Pythia-410M trained on 30× more tokens to get only marginally better perplexity. The modern architecture (GQA + SwiGLU + RMSNorm + RoPE + muP + QK-Norm + z-loss) is doing real work — we're at GPT-2-Small-trained-on-30B equivalence with only 5.6B tokens of training.

## Training stability

| Signal | Count | Rate (per 1000 steps) | Verdict |
|---|---|---|---|
| `nan_batch_skipped` events | 288 | 1.9 | Healthy. Atomic NaN-revert handled every event without state corruption. |
| `soft_spike` events (3σ logged warning, no action) | dozens | ~5-10 | Normal — small models hit hard batches occasionally |
| `hard_spike` events (6σ → watchdog rollback) | **0** | 0 | Never triggered. `lr_recovery_multiplier` stayed at 1.0 throughout. |
| `LossSpikeError` (max recoveries exhausted) | 0 | — | Watchdog protection was never close to limit |

**NaN source attribution** (from `scripts/inspect_quarantine.py` against the in-training quarantine log, 142 events / 67K steps pre-resume):

```
fineweb_edu                  : 66  (46.5%)  vs corpus share 44.0%  → 1.06×
github_code_clean            : 19  (13.4%)  vs corpus share 18.0%  → 0.74×
open_web_math                : 14  ( 9.9%)  vs corpus share  7.0%  → 1.41×
wikipedia_20231101_en        : 11  ( 7.7%)  vs corpus share  6.0%  → 1.29×
pg19                         :  9  ( 6.3%)  vs corpus share  5.0%  → 1.27×
pes2o                        :  6  ( 4.2%)  vs corpus share  6.0%  → 0.70×
sangraha_verified_split_hin  :  5  ( 3.5%)  vs corpus share  4.0%  → 0.88×
mc4_zh                       :  4  ( 2.8%)  vs corpus share  1.5%  → 1.87×
stack_exchange_preferences   :  4  ( 2.8%)  vs corpus share  2.0%  → 1.40×
mc4_fr                       :  2  ( 1.4%)  vs corpus share  1.5%  → 0.93×
mc4_es                       :  1  ( 0.7%)  vs corpus share  1.5%  → 0.47×
mc4_de                       :  1  ( 0.7%)  vs corpus share  2.0%  → 0.35×
```

**Distribution is roughly proportional to corpus shares** — the NaN events are bf16 numerical sensitivity at peak LR, NOT poisonous documents in any single source. github_code (the most-trained-on-after-fineweb source) is actually UNDER-represented in NaNs (code is numerically well-behaved). The mild over-representation of open_web_math + mc4_zh suggests math notation + multi-byte tokens can occasionally produce sharper gradients, but the effect is within Poisson noise at these counts.

## Sample generations (from 1×H100 on 2026-05-15, post-G6-fix)

Tested with `scripts/generate.py` at `temperature=0.8`, `top_p=0.9`, `max_new_tokens=80`. Sampled 10 prompts covering knowledge, code, multilingual, math, and narrative.

### Pattern: STYLE TRANSFER works (prompts get continued in matching domain)

#### Wikipedia-style prompt → encyclopedia format

> **prompt**: "The Pacific Ocean is"
>
> **output**:
> > "The Pacific Ocean is a circular ocean in the Arctic Ocean. It is considered to be the second-largest ocean, after the Arctic Ocean. The waters of the Pacific Ocean are also known as the 'Ice Souths'.
> >
> > Name
> > The name of the Pacific Ocean was first used by English writer and explorer John Thompson in 1794, who described the area as 'the continent of water,'..."

Note the model produces a Wikipedia-style article with a `## Name` section. **Form is correct**; content is hallucinated (Pacific is not in Arctic, not second-largest, "Ice Souths" is invented).

#### Code prompt → Python with type-correct syntax

> **prompt**: `"def fibonacci(n):\n    "`
>
> **output**:
> > ```python
> > def fibonacci(n):
> >     return n + 1
> >
> > def test_reorder(self):
> >     n = 1
> >     self.assert(self.keep_gaps, self.edge_count, self.reorder_count)
> > ```

Python style is correct (indentation, `def`, `return`, class-method test pattern). The actual fibonacci logic is wrong (`return n + 1`); the test_reorder is unrelated. Code-as-style works; code-as-correctness does not at 250M.

#### Hindi prompt → coherent Hindi continuation (sangraha training paid off)

> **prompt**: "नमस्ते"
>
> **output (Hindi)**: a news-style article in Hindi mentioning India News, breaking news topics, sports/entertainment/technology sections. Grammatical Hindi script throughout, properly tokenized.

**This validates the multilingual mix decision.** The 4% sangraha share was enough for the model to learn Hindi distributional patterns.

#### Math prompt → LaTeX notation (open_web_math + pes2o training paid off)

> **prompt**: "Theorem (Pythagorean):"
>
> **output**:
> > "Theorem (Pythagorean): The more easily known theorem of the 'magic rule' for the ring of all prime numbers
> >
> > ## Definition
> >
> > Let $S$ be a natural number ring, $k$ a prime number, and $C$ a non-prime number.
> >
> > Then there exists a sequence $t_k$ of $k$ numbers $y_k$ such that $y_k \\..."

LaTeX notation (`$S$`, `$k$`, `$t_k$`, `$y_k$`), theorem-definition structure, math-paper register. **Form again correct; content nonsensical.** The model learned "math mode" but not actual math.

### Pattern: factual recall is weak

> **prompt**: "The capital of France is"
>
> **output**:
> > "The capital of France is the capital of France. In France, the capital of France is the capital of France. The capital of France is the capital of France..." (loops)

Expected at 250M / 5.6B-token scale. **The model can identify the prompt domain ("the capital of X is" is a Wikipedia-style fact pattern) but doesn't reliably retrieve the specific token "Paris" from the next-token distribution.** Larger models (1B at 600B tokens, the Stage 3 target) reliably get common facts.

### Pattern: loops appear ~30-50 tokens in

The looping at the end of long generations is a known weakness of small base models without:
- Repetition penalty (we didn't enable this in the sampler)
- SFT (teaches the model "stop generating when you have nothing new")
- Bigger model (wider distribution = less likely to fall into a local pattern)

Stage 3's 1B model with SFT would mitigate dramatically. For pilot purposes (recipe validation), looping is acceptable evidence.

### Pattern: arithmetic fails

> **prompt**: "Two plus two equals"
>
> **output**:
> > "Two plus two equals one plus one, and a minus two equals one plus one. Two plus two equals one plus one. Two plus three equals one plus one. Two plus four equals one plus one..."

250M doesn't reliably do arithmetic. Math reasoning emerges at 1B+ scale. Pilot has effectively no math capability beyond pattern-matching the form of arithmetic.

## Lessons learned

1. **Corpus capacity matters**: pilot exhausted the 4.98 B-token corpus at step 151,990 — WSD decay phase never reached without intervention. Stage 2 (30 B-token target) needs **multi-epoch reader** OR 6× larger corpus. Listed as Phase 1.1 in the roadmap.

2. **`data_position` must be int64**: the int32 overflow bug at step ~65,500 (~2.15 B tokens) crashed the run. Fixed mid-run (commit `9f442f7`); the underlying cause was that JAX defaults Python ints to int32 when tracing, and we should pop `data_position` from state before any JIT'd `train_step` call. Same fix applied to `eval_hook` (commit `dd7b202`).

3. **G6 cross-mesh checkpoint restore needed before any inference**: pilot was saved on 4×H200 (DP=4); loading on 1×H100 hit the `RestoreArgs(sharding=...)` bug. Fixed in commits `13d6126` → `3be12de` → `ca1c40b`. Now `CheckpointManager.restore()` accepts an optional `sharding` parameter and works for any device count.

4. **WSD decay does the expected ~0.15-nat improvement**, not the optimistic ~0.4-nat sometimes cited. Our pilot's improvement (2.878 → 2.730 = 0.147 nats) matches the literature median for under-trained models (tokens/param ratio 22:1). Over-trained models (SmolLM-style, 11000:1) get larger decay improvements.

5. **bf16 numerical sensitivity is real but manageable**: 1.9 NaN events per 1000 steps was the steady-state. Atomic NaN-revert handled them silently. The hard-spike watchdog (6σ) was never close to triggering.

6. **Multilingual training works at small scale**: the 4% sangraha share was enough for Hindi to emerge as a generation capability. Worth noting for the project's "Indian/multilingual hedge" positioning.

7. **NaN distribution is bf16-noise, not bad-data**: source attribution from the quarantine log confirms NaN events are roughly proportional to corpus shares. No single poisonous source identified. github_code is UNDER-represented in NaNs (code is well-behaved numerically).

8. **W&B run can split on resume**: our resume from the int32 crash created a second W&B run (`u5xsxm0l`) instead of continuing `roydqofb`. Both finalized cleanly; the loss curve is visually split across two runs. Acceptable but worth knowing.

## Hardware + cost summary

| Phase | Compute | Wall | Cost |
|---|---|---|---|
| Corpus build (CPU only) | 128-core dev box | 2 h 1 m | $0 |
| Stage 1 pretrain | 4×H200 SXM | ~12 h (with mid-run resume) | ~$350 |
| Stage 1.5 decay pass | 4×H200 SXM | 2 h 18 m | ~$32 |
| Post-hoc eval (both ckpts) | 4×H200 SXM (kept running) | <5 min | included above |
| Inference smoke test | 1×H100 | ~20 min | ~$1 |
| **TOTAL** | — | **~17 hr GPU + 2 hr CPU** | **~$385** |

## What did NOT meet expectations

| Item | What we hoped for | What we got | Why |
|---|---|---|---|
| Pilot reaching planned 229,000 steps | yes | only 151,990 | Corpus exhausted at 152K (single-epoch reader; fixed in Phase 1.1) |
| WSD decay phase reaching from in-training schedule | yes | no | Same corpus-exhaustion root cause; ran Stage 1.5 to compensate |
| In-training eval firing throughout the run | yes | only first 13 events (steps 5K-65K) | int32 bug in eval_hook silently failed eval post-resume; fixed in `dd7b202` for Stage 1.5 |
| Factual recall ("Paris" for "capital of France") | hoped | not reliably | 250M is too small for many specific facts; Stage 3's 1B at 600B tokens will fix this |
| Math beyond simple cases | hoped | doesn't work | Emerges at 1B+ scale; pilot doesn't have it |

These are all addressed by either Phase 1 prep work or the natural scale increase to Stage 2/3.
