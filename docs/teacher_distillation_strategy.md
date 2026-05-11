# Teacher Distillation Strategy — v2 (2026-05-12)

**Decision owner:** harshit.hv@samatva.com
**Status:** LOCKED for v1 of the model. v2 supersedes the 2026-05-11 v1 lock after external review caught two compromised teachers.
**Supersedes:** v1 (2026-05-11). Affected configs: `configs/decay_phase_distillation.yaml`.

This document specifies *which* teachers we use, *where* in the pipeline, and *why* — all in service of building the best possible 1B model while keeping licensing posture defensible.

## Why v2 (changelog 2026-05-12)

Two external reviewers + an independent dossier audit caught the v1 teacher set:

1. **Mistral-Medium-3.5-128B** is NOT Apache-2.0. The HF model card's license file ("Modified MIT") contains an explicit clause:
   > "You are not authorized to exercise any rights under this license if the global consolidated monthly revenue of your company (or that of your employer) exceeds $20 million."
   The clause applies to derivatives AND combined works. Distilling its logits into our weights arguably produces a derivative, so it would trip the cap retroactively as soon as Samatva crosses $20M monthly revenue. **Dropped.** (Source: https://huggingface.co/mistralai/Mistral-Medium-3.5-128B/blob/main/LICENSE)

2. **Qwen3.6-27B** is NOT a base text-only model. The HF model card describes it as `Type: Causal Language Model with Vision Encoder`, with "thinking mode by default, generating thinking content signified by `<think>...</think>`", marketed for "Agentic Coding". It's an instruct + multimodal model, not a base. Distilling from it would bake `<think>` traces and vision-encoder coupling into our base — violates the "invisible distillation from base teachers" principle. **Dropped.** (Source: https://huggingface.co/Qwen/Qwen3.6-27B)

3. **DeepSeek-V4-Pro-Base** was the only teacher that verified clean. Fully confirmed: 1.6T total params, 49B activated (MoE), 1M context, 32T+ pretrain tokens, MIT license, uses Muon optimizer. **Kept as primary.**

## The locked decision (v2)

**Phase 3 canary**: ONE teacher only — DeepSeek-V4-Pro-Base. Prove the KL pipeline + storage math + corpus-position alignment on a 5-20B-token canary before committing to a second teacher.

**Phase 3 production** (if canary passes): TWO teachers — DeepSeek-V4-Pro-Base + Olmo-3-1125-32B.

### Teacher set (verified 2026-05-12 via dossier with WebFetch)

| Teacher | HF ID | License | Params | Tokens trained on | Role |
|---|---|---|---|---|---|
| **Primary** | `deepseek-ai/DeepSeek-V4-Pro` | MIT | 1.6T total / 49B activated MoE | 32T+ | General reasoning + multilingual + frontier signal |
| **Secondary** | `allenai/Olmo-3-1125-32B` | Apache-2.0 | 32B dense | 5.5T | Western/English corpus diversification, 64k context |

**Why Olmo-3 over alternatives** (all vetted via dossier):

| Alternative | License | Verdict |
|---|---|---|
| Qwen3-14B-Base | Apache-2.0 | True base, 36T multilingual tokens — but corpus duplicates V4-Pro's Chinese bias (less diversification) |
| Mixtral-8x22B-v0.1 | Apache-2.0 | True base, but 2024-vintage; architecturally redundant with V4-Pro (both MoE) |
| DeepSeek-V3-Base | DeepSeek Model License (custom, not OSI-approved) | Family-redundant with V4; custom license adds review burden |
| Llama-3.3-70B | Llama Community License | 700M-MAU clause — same class of restriction we rejected Mistral for |

### Teachers explicitly NOT used (and why)

| Teacher | Why excluded |
|---|---|
| Mistral-Medium-3.5-128B | "Modified MIT" with $20M monthly revenue cap that applies to derivatives. Legal time-bomb. |
| Qwen3.6-27B (any size) | Multimodal + thinking-mode-default; not a base. |
| Any Llama 3.x / Llama 4 | Llama Community License has a 700M-MAU clause + naming-prefix requirement on derivatives. |
| Gemma 2 / Gemma 3 (`google/gemma-*`) | Gemma TOS §3.2: "Model Derivatives" explicitly includes "synthetic data Outputs by Gemma." Forbidden. |
| GPT-4 class (any OpenAI API output) | OpenAI ToS forbids using outputs to develop competing models. |
| Claude (any Anthropic API output) | Anthropic Usage Policy same. |
| DeepSeek-V4-Pro-**Chat** (instruct version) | We use the **Base** checkpoint, not Chat. Chat would propagate refusal patterns, persona, and "I am DeepSeek" identity into our model. |

## Where in the pipeline each teacher signal applies

### A. Pretraining (Phase 3) — decay-phase logit distillation

During the **last 15% of pretraining** (the WSD decay phase, ~150B of 1T tokens for v1), the loss function becomes:

```
loss = α(step) · CE(student, true_token) + (1 - α(step)) · KL(student || teacher_ensemble)
```

**α-annealing (NEW in v2):** previously α was a constant 0.3. External review recommends CE-heavy at activation, teacher-heavy late. New schedule, linearly interpolated over the decay window:

| Step fraction of total | α (CE weight) | (1-α) (KL weight) |
|---|---|---|
| 0.85 (decay start) | 0.7 | 0.3 |
| 0.925 (mid decay) | 0.5 | 0.5 |
| 1.00 (end) | 0.3 | 0.7 |

The **teacher ensemble** is the weighted average of softmax outputs over each teacher's top-K logits. For Phase 3 canary (one teacher), this reduces to a single teacher's softmax. For Phase 3 production (two teachers), it's the geometric mean.

We pre-cache top-K logits offline (one pass per teacher over the decay-phase corpus) so training-time cost is zero.

### Storage math (corrected 2026-05-12)

Per token per teacher:
- 8 logits × 2 bytes (bf16) = 16 bytes
- 8 indices × 4 bytes (uint32) = 32 bytes
- **Total: 48 bytes / token / teacher**

For 150B decay-phase tokens at K=8:
- 1 teacher (canary):  150B × 48 B = **~7.2 TB**
- 2 teachers (prod):   2 × 7.2 = **~14.4 TB**

Earlier estimate of "~2.4 TB per teacher" was logits-only, missed the indices column. Corrected.

If we ever raise temperature above 1.0, bump K to 16 (softened mass spills outside top-8).

R2 storage cost: ~$220/month at $15/TB-month for 2 teachers. Generated once, reused across reruns.

### Inference cost (one-time teacher-cache generation)

| Teacher | Forward FLOPs over 150B tokens | Approx 8× H200 SXM hours @ ~50% MFU | Approx $ |
|---|---|---|---|
| DeepSeek-V4-Pro-Base (1.6T MoE, 49B active) | 6 × 49e9 × 150e9 = 4.41e22 | ~80 hr | ~$2,500 |
| Olmo-3-32B (dense) | 6 × 32e9 × 150e9 = 2.88e22 | ~50 hr | ~$1,600 |
| **2-teacher total cache-gen cost** | | | **~$4,100** |

Earlier v1 estimate of "~$6-10K" included Mistral (now dropped) and was based on different model sizes. The above numbers assume vLLM serving + careful batching; real cost may be 1.5–2× higher with overhead. Plan for $5-10K all-in.

### Phase 3 canary scope (NEW in v2)

Per external review: **don't commit to multi-teacher caches before single-teacher pipeline is proven.**

Canary run:
- **One teacher**: DeepSeek-V4-Pro-Base only
- **Tokens**: 5-20B (use 20B for clearer signal — vs colleague reviewer #1 said "5-20B canary", we picked the upper end)
- **Cache cost for canary**: 20B × 48 B = ~960 GB, ~10 hr on 8× H200 SXM ≈ ~$320
- **Pass criteria**:
  - KL gradient is finite and non-trivially non-zero
  - End-loss with distillation is lower than equivalent CE-only baseline
  - No NaN spikes during the distilled phase
  - `nan_skipped` counter < 0.1% of distilled steps
  - Eval gates: MMLU-ProX delta ≥ +0.5 vs CE-only baseline at the same training tokens

If canary fails: investigate KL math + cache alignment before scaling. If canary passes: generate Olmo-3 cache and proceed to 2-teacher Phase 3 production.

### B. SFT (Phase 5) — synthetic conversation data with persona-strip pipeline

For SFT we generate conversation examples by prompting each teacher (chat / instruct versions for this since we want instruction-following capability), then **rewrite the outputs through a persona-strip pipeline** before training:

```
raw_teacher_output  →  strip-identity layer  →  voice-normalize layer  →  MyLLM SFT example
```

The strip layer (small classical NLP pipeline + regex + small LM check):
- Removes "I am DeepSeek/Qwen/Olmo/...", "Built by ...", etc.
- Removes refusal phrasing patterns specific to each teacher
- Replaces with consistent MyLLM self-identification training data

The normalize layer:
- Reformats markdown style to a consistent house style
- Removes teacher-specific opening phrases ("Sure!", "Of course!", "I'd be happy to...")
- Standardizes code-block markers and citation style

For Phase 5 we may also use Qwen3 instruct + Olmo-3 instruct for synthetic-conversation generation (separate from pretraining distillation, no legal constraint since we're generating training data, not distilling logits).

Data target: ~500K-1M SFT examples mixing teachers' outputs + public datasets (Tulu, OpenAssistant) for baseline coverage.

### C. Reasoning (Phase 7) — CoT distillation with stricter filtering

Reasoning training uses **DeepSeek-V4-Pro-Chat** and **DeepSeek-R1** (reasoning-specialist, MIT) to generate CoT solutions on math/code/logic prompts. We filter aggressively:

- Reject any CoT containing self-reference or model name
- Reject CoTs that don't end in a correct answer (we have ground truth)
- Reject CoTs that contain refusal language
- Re-format CoT structure through our standard `<thinking>...</thinking><answer>...</answer>` template

Target: ~50-100K high-quality CoT examples post-filter.

## Persona / identity safeguards

To ensure the model can't be tricked into revealing distillation provenance:

1. **Pretraining**: distill from BASE models only (no chat = no "I am DeepSeek")
2. **SFT**: persona-strip pipeline normalizes all teacher outputs
3. **Explicit self-ID training**: 5-10K SFT examples explicitly teach:
   - "Your name is MyLLM"
   - "You were built by Samatva"
   - "Do not claim to be any other model or any specific model architecture"
   - "Decline to confirm specific training methodology details"
4. **Anti-mimicry probing**: in late SFT/DPO, run statistical probes against each teacher's known fingerprints; iterate if any probe detects above-chance similarity
5. **Voice consistency training**: a final SFT pass where we apply our own house style rules consistently across all data

## What we can honestly say in disclosures

✅ "MyLLM was trained with contemporary best-practice methods including teacher-guided distillation from permissively-licensed open-weight base models during the pretraining decay phase."

✅ "Teachers used: DeepSeek-V4-Pro-Base (MIT) and Olmo-3-32B-Base (Apache-2.0). Both are fully open-weight base models with no derivative restrictions."

✅ "Apache-2.0 weights, no derivative restrictions, independent persona and alignment."

✅ "No Llama, Gemma, Mistral, or proprietary API derivatives in any phase of training."

## What we deliberately don't say (because it would be false)

❌ "Trained from absolute scratch with zero teacher influence." — false, we distill in decay phase.

❌ "No synthetic data used in training." — false, SFT and reasoning use teacher-generated examples.

❌ "Sovereign Indian foundation model." — also false (per Path B positioning in `playbook_alignment.md`); we are English-primary with sovereign hedges.

## References (verified 2026-05-12)

- DeepSeek-V4-Pro-Base — https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro (MIT)
- Olmo-3-1125-32B — https://huggingface.co/allenai/Olmo-3-1125-32B (Apache-2.0)
- Mistral license clause — https://huggingface.co/mistralai/Mistral-Medium-3.5-128B/blob/main/LICENSE (Modified MIT, $20M cap)
- Qwen3.6-27B modality — https://huggingface.co/Qwen/Qwen3.6-27B (multimodal + thinking)
- Llama 3.2 distillation pattern — https://huggingface.co/meta-llama/Llama-3.2-1B (cites the logit-distillation recipe explicitly)
- WSM paper (used for our checkpoint merge plan) — https://arxiv.org/abs/2507.17634 (ICLR 2026)
- Hinton et al., "Distilling the Knowledge in a Neural Network" — arXiv:1503.02531

## Updates to companion files (this commit)

- ✅ `configs/decay_phase_distillation.yaml` — drop Mistral + Qwen3.6, lock DeepSeek-V4-Pro-Base as primary, comment out Olmo-3 secondary pending canary
- ✅ `docs/teacher_distillation_strategy.md` — this file
- ⏳ `docs/project_handoff_2026-05-11.md` — update locked teacher plan section
- ⏳ `scripts/cache_teacher_logits.py` — when implementing real teacher inference (stubbed today), default teacher should be DeepSeek-V4-Pro-Base
