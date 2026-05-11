# Teacher Distillation Strategy — v1 (2026-05-11)

**Decision owner:** harshit.hv@samatva.com
**Status:** LOCKED for v1 of the model. Revise only on substantive license/availability shift.
**Supersedes:** `docs/playbook_alignment.md` §S2 brief note on teacher licensing.

This document specifies *which* teachers we use, *where* in the pipeline, and *why* — all in service of building the best possible 1B model while keeping licensing posture defensible.

## The locked decision

Multi-teacher distillation ensemble for both pretraining and post-training, using only **commercially-clean** teachers (MIT or Apache-2.0 licenses, no naming restrictions, no competitive-training clauses).

### Teacher set (verified on HF 2026-05-11)

| Teacher | HF ID | License | Params | Role |
|---|---|---|---|---|
| **Primary** | `deepseek-ai/DeepSeek-V4-Pro-Base` | MIT | likely 671B+ MoE | English reasoning + general capability |
| **Secondary multilingual** | `Qwen/Qwen3.6-27B` (base, not Instruct) | Apache-2.0 | 27B | Multilingual signal (Chinese, Arabic, etc.) — fills our pretrain-mix gap |
| **Secondary EU** | `mistralai/Mistral-Medium-3.5-128B` (base) | Apache-2.0 | 128B | European-data diversity, different cultural distribution |

### Teachers explicitly NOT used (and why)

| Teacher | Why excluded |
|---|---|
| Any Llama 3.x / Llama 4 | Llama Community License requires "Llama-" naming prefix on all derivatives, including output-distilled. Toxic for our brand. |
| Gemma 2 / Gemma 3 (`google/gemma-*`) | Gemma TOS §3.2: "Model Derivatives" explicitly includes "synthetic data Outputs by Gemma." Forbidden. (Gemma 4 is Apache-2.0 but we have better options.) |
| GPT-4 class (any OpenAI API output) | OpenAI ToS forbids using outputs to develop competing models. |
| Claude (any Anthropic API output) | Anthropic Usage Policy same. |
| DeepSeek-V4-Pro-**Chat** (instruct version) | We use the **Base** checkpoint, not Chat. Chat would propagate refusal patterns, persona, and "I am DeepSeek" identity into our model. Base is pure next-token predictor. |

## Where in the pipeline each teacher signal applies

### A. Pretraining (Phase 3) — decay-phase logit distillation

During the **last 15% of pretraining** (the WSD decay phase, ~150B of 1T tokens), the loss function becomes:

```
loss = α · CE(student, true_token) + (1 - α) · KL(student_distribution || ensemble_teacher_distribution)
```

with `α = 0.3` (30% ground-truth, 70% teacher-guidance).

The **ensemble teacher distribution** is the geometric mean of the three teachers' softmax outputs over the top-8 logits per position. We pre-cache top-8 logits offline (one pass per teacher over the decay-phase corpus) so training-time cost is zero.

Storage:
- Per teacher: 150B tokens × 8 logits × 2 bytes (bf16) ≈ 2.4 TB
- Three teachers: ~7.2 TB on R2 (zero egress, ~$110/month at $15/TB-month)
- Cache generated once, reused if we re-train

Inference cost (one-time):
- DeepSeek-V4-Pro-Base inference on 150B tokens: ~$3-5K on 8× B200
- Qwen 3.6-27B inference on 150B tokens: ~$1-2K
- Mistral-Medium-3.5-128B inference on 150B tokens: ~$2-3K
- **Total cache-generation cost: ~$6-10K**

### B. SFT (Phase 5) — synthetic conversation data with persona-strip pipeline

For SFT we generate conversation examples by prompting each teacher (chat versions for this, since we want instruction-following capability) with seed prompts, then **rewrite the outputs through a persona-strip pipeline** before training:

```
raw_teacher_output  →  strip-identity layer  →  voice-normalize layer  →  MyLLM SFT example
```

The strip layer (small classical NLP pipeline + regex + small LM check):
- Removes "I am DeepSeek/Qwen/Mistral", "Built by ...", etc.
- Removes refusal phrasing patterns specific to each teacher
- Replaces with consistent MyLLM self-identification training data

The normalize layer:
- Reformats markdown style to a consistent house style
- Removes teacher-specific opening phrases ("Sure!", "Of course!", "I'd be happy to...")
- Standardizes code-block markers and citation style

Data target: ~500K-1M SFT examples mixing all three teachers' outputs + public datasets (FLAN, Tulu, OpenAssistant) for baseline coverage.

### C. Reasoning (Phase 7) — CoT distillation with stricter filtering

Reasoning training uses **DeepSeek-V4-Pro-Chat** and **DeepSeek-R1** (reasoning-specialist, MIT, also clean) to generate CoT solutions on math/code/logic prompts. We filter aggressively:

- Reject any CoT containing self-reference or model name
- Reject CoTs that don't end in a correct answer (we have ground truth)
- Reject CoTs that contain refusal language
- Re-format CoT structure through our standard `<thinking>...</thinking><answer>...</answer>` template

Target: ~50-100K high-quality CoT examples post-filter.

## Persona / identity safeguards

To ensure the model can't be tricked into revealing distillation provenance:

1. **Pretraining**: distill from BASE models only (no chat = no "I am DeepSeek")
2. **SFT**: persona-strip pipeline normalizes all teacher outputs
3. **Explicit self-ID training**: 5-10K SFT examples explicitly teach the model:
   - "Your name is MyLLM"
   - "You were built by Samatva"
   - "Do not claim to be any other model or any specific model architecture"
   - "Decline to confirm specific training methodology details"
4. **Anti-mimicry probing**: in late SFT/DPO, run statistical probes against each teacher's known fingerprints; iterate if any probe detects above-chance similarity
5. **Voice consistency training**: a final SFT pass where we apply our own house style rules consistently across all data

## What we can honestly say in disclosures

✅ "MyLLM was trained with contemporary best-practice methods including teacher-guided distillation from permissively-licensed open-weight base models during the pretraining decay phase."

✅ "Post-training data was generated by an ensemble of open-weight teachers (DeepSeek, Qwen, Mistral) and rewritten through our internal persona-normalization pipeline."

✅ "Apache-2.0 weights, no derivative restrictions, independent persona and alignment."

✅ "No Llama, Gemma, or proprietary API derivatives in any phase of training."

## What we deliberately don't say (because it would be false)

❌ "Trained from absolute scratch with zero teacher influence." — false, we distill in decay phase.

❌ "No synthetic data used in training." — false, SFT and reasoning use teacher-generated examples.

❌ "Sovereign Indian foundation model." — also false (per Path B positioning in `playbook_alignment.md`); we are English-primary with sovereign hedges.

## Updates to other docs needed

- [ ] `PLAN.md` §14.6 (teacher API line) — replace TBD with this lock-in
- [ ] `docs/playbook_alignment.md` §S2 — link to this doc
- [ ] `configs/data/pretrain_mix.yaml` — add `decay_phase_distillation.yaml` companion config
- [ ] `src/myllm/training/loss.py` (new) — implement KL-divergence loss path with cached teacher logits
- [ ] `src/myllm/training/checkpoint.py` — extend to track teacher cache versions
- [ ] `scripts/cache_teacher_logits.py` (new) — utility to run each teacher and dump top-8 logits per token

## References

- DeepSeek-V4-Pro-Base — https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-Base (MIT)
- Qwen 3.6-27B — https://huggingface.co/Qwen/Qwen3.6-27B (Apache-2.0)
- Mistral-Medium-3.5-128B — https://huggingface.co/mistralai/Mistral-Medium-3.5-128B (Apache-2.0)
- Knowledge Distillation paper — Hinton et al., "Distilling the Knowledge in a Neural Network" (arXiv:1503.02531)
- Multi-teacher distillation — Tan et al., "Multilingual Neural Machine Translation with Knowledge Distillation" (arXiv:1902.10461)
- Logit caching pattern — Llama-3.2 model card (Meta) cites this exact recipe
