# External Technical Review — Enterprise 2026 Research Review
*Received 2026-05-12 (separate reviewer from `MyLLM_Repo_Technical_Review_2026-05-12.docx`)*
*Source: `MyLLM_Enterprise_2026_Research_Review.pdf` (provided in handoff conversation; binary not committed)*

This is a markdown summary of the second external review of the MyLLM project. The original PDF was provided in the handoff conversation. Key claims have been independently verified via dossier WebFetch on 2026-05-12 (verification results inline below).

## Bottom-line verdict

> "Your overall direction is good: a dense 1B-class decoder-only Transformer, strong tokenizer, English-primary plus Hindi/Indic hedge, WSD/WSM, muP, decontamination, and evaluation gates are all defensible. The bigger issue is that the 2026 small-model frontier has moved toward highly engineered data, multi-trillion-token training, distillation, reasoning-aware post-training, and enterprise governance."

> "Treat 1T tokens as an internal v1 checkpoint, not the final enterprise release."

## Critical claims verified by dossier (2026-05-12)

### Teacher licensing

| Teacher | Reviewer's claim | Verification result | URL |
|---|---|---|---|
| DeepSeek-V4-Pro-Base | MIT, 1.6T params, 49B activated MoE, 1M context, 32T+ tokens, uses Muon | ✅ FULLY CONFIRMED | https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro |
| Qwen3.6-27B | Multimodal causal LM with vision encoder, thinking-mode-default — NOT a base text teacher | ✅ CONFIRMED | https://huggingface.co/Qwen/Qwen3.6-27B |
| Mistral-Medium-3.5-128B | "Modified MIT" license with $20M monthly revenue cap on derivatives — NOT Apache-2.0 | ✅ CONFIRMED | https://huggingface.co/mistralai/Mistral-Medium-3.5-128B/blob/main/LICENSE |

**Action taken**: Mistral + Qwen3.6 dropped from locked teacher plan. DeepSeek-V4-Pro-Base kept as primary. Olmo-3-1125-32B (Apache-2.0, 32B dense, 5.5T tokens, 64k context) added as secondary. See `docs/teacher_distillation_strategy.md` v2 for full rationale.

### Frontier token budgets (1B-class models)

| Model | Reviewer's claim | Verification | URL |
|---|---|---|---|
| Llama 3.2 1B/3B | Up to 9T tokens, used logit distillation from Llama 3.1 8B/70B | ✅ CONFIRMED | https://huggingface.co/meta-llama/Llama-3.2-1B |
| OLMo 2 1B | 4T tokens | ✅ CONFIRMED | https://huggingface.co/allenai/OLMo-2-0425-1B |
| SmolLM3 3B | 11.2T tokens with staged curriculum | ✅ CONFIRMED | https://huggingface.co/HuggingFaceTB/SmolLM3-3B |

**Action taken**: Our 1T plan is reframed as "internal v1 / stack validation" not "external 1B release". Continuation path to 3T planned for after v1 evals.

### Datasets

| Dataset | Reviewer's claim | Verification | URL |
|---|---|---|---|
| Sangraha | 251B tokens / 22 languages / CC-BY-4.0 / Hindi 34.5B | ✅ FULLY CONFIRMED | https://huggingface.co/datasets/ai4bharat/sangraha |
| Nemotron-CC | 6.3T tokens (4.4T real + 1.9T synthetic), stronger MMLU vs DCLM | ❓ COULDN'T VERIFY (dataset is HF-gated; returns 401) | https://huggingface.co/datasets/nvidia/Nemotron-CC |

### Methodology / papers

| Claim | Verification | URL |
|---|---|---|
| WSM paper (ICLR 2026): +3.5% MATH, +2.9% HumanEval, +5.5% MMLU-Pro over WSD; merge duration > checkpoint count | ✅ CONFIRMED (reviewer's arxiv ID was wrong — correct is **arxiv:2507.17634** not 2410.05192) | https://openreview.net/forum?id=HhThhjKyfw |
| EU AI Act GPAI obligations effective 2 August 2025 | ✅ CONFIRMED — tech doc + copyright policy + training-content summary required | https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai |

## Key strategic recommendations adopted

### 1. Token budget reframing (adopted)
1T = internal v1 only. Plan continuation to 3T after v1 evals if eval curves are healthy.

### 2. Proxy hierarchy (adopted)
- Proxy A (existing): 67M, 2k context — cheap LR/init sweep
- **Proxy B (NEW)**: 300-500M, 4k context, same GQA ratio as base — transfer validation before pilot

### 3. Distillation canary scope (adopted)
1 teacher (DeepSeek-V4-Pro-Base) × 5-20B tokens before committing to multi-teacher 150B+ cache.

### 4. α-annealing (adopted)
Replace constant α=0.3 with linear schedule 0.7 (start of decay) → 0.5 (mid) → 0.3 (end).

### 5. K=16 if temperature > 1 (acknowledged; we keep T=1.0 so K=8 stays valid)

### 6. Storage math correction (acknowledged)
Per token per teacher = 8 × (2 byte logit + 4 byte index) = **48 bytes**, not the 16 bytes we'd previously assumed (which was logits-only). For 2-teacher × 150B tokens: ~14.4 TB.

### 7. Evaluation gate extensions (queued for Phase B)
- LiveCodeBench (contamination-free coding)
- RULER (long context, replaces NIAH)
- MMLU-Pro (harder MMLU variant)
- MGSM (multilingual math)
- BFCL for tool calling
- Calibration: ECE
- Toxicity/bias: multilingual safety sets
- Data leakage: canary strings + memorization probes

### 8. Enterprise governance (queued as Phase 5)
- NIST GenAI Profile for AI RMF
- ISO/IEC 42001
- OWASP GenAI Security + Agentic Top 10
- EU AI Act GPAI artifacts (mandatory)
- India MeitY AI Governance Guidelines + DPDP framework
- Model card, data card, eval card, risk card

### 9. Serving / quantization (queued as Phase 6)
- BF16 (training) / FP8 / INT8 / INT4 / GGUF targets
- TensorRT-LLM (production GPU) / vLLM (experimentation) / llama.cpp-compatible (on-device)
- Always evaluate quantized models separately

## Confirmed Phase A gates (overlap with reviewer's "Phase 0 - harden the repo")

| Reviewer's P0 gate | Our Phase A fix |
|---|---|
| True NaN skip (params + opt_state unchanged) | ✅ A2 (atomic `jnp.where`) |
| End-to-end doc masking | ✅ A4 (`segment_ids` plumbed) |
| Cross-document loss masking | ✅ A4 (`loss_mask`) |
| Token accounting | ✅ A3 (`model_cfg.context_length` authoritative) |
| Checkpoint + data cursor resume | ✅ A6 (`data_position` persisted) |
| Bf16 cache decoding | ✅ A1 (`_bf16_bits_to_f32` at reader boundary) |
| Sample by token share | ✅ A5 (deficit-driven mixture) |
| muP opt_state restore | ⏳ Phase B (MultiTransformState Orbax) |
| Deterministic packed corpus | ⏳ Phase B (offline corpus + manifest) |
| Batch provenance dump | ⏳ Phase B (quarantine file) |

## Decisions made by harshit (2026-05-12 follow-up)

1. **Token budget**: 1T first, then higher T if v1 evals are healthy — confirmed
2. **Hindi share**: keep at 4% — confirmed (reviewer suggested 8-12% if Hindi generation is a real product target; harshit's call is 4% is enough)
3. **Proxy B (300M intermediate)**: yes — adopt
4. **Canary scope**: "whatever is better" — interpreted as 20B-token canary (upper end) for clearer signal
5. **Mistral**: drop, replace if possible — done; Olmo-3-32B-Base selected
6. **Canary teacher count**: 1 (DeepSeek-V4-Pro-Base only) — confirmed
7. **2nd-teacher search**: yes — done; Olmo-3-1125-32B selected
8. **Commit reviews + update configs/docs**: yes — done in this commit

## What this reviewer's frame adds beyond the colleague's review

The first reviewer (`MyLLM_Repo_Technical_Review_2026-05-12.docx`) focused on **code correctness**: 6 P0 silent-corruption bugs in the training pipeline. We fixed all 6 in Phase A.

This second reviewer focused on **enterprise strategy**: governance, serving economics, regulatory compliance (EU AI Act, India DPDP/MeitY), and explicit comparison against the 2026 frontier (Llama 3.2 / OLMo 2 / SmolLM3 / Qwen3). They're complementary; together they constitute the full pre-Phase-2 audit.

## References

Primary HF model cards verified 2026-05-12:
- https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro
- https://huggingface.co/allenai/Olmo-3-1125-32B
- https://huggingface.co/mistralai/Mistral-Medium-3.5-128B/blob/main/LICENSE
- https://huggingface.co/Qwen/Qwen3.6-27B
- https://huggingface.co/Qwen/Qwen3-14B-Base (rejected secondary candidate)
- https://huggingface.co/mistralai/Mixtral-8x22B-v0.1 (rejected secondary candidate)
- https://huggingface.co/deepseek-ai/DeepSeek-V3-Base (rejected secondary candidate — custom license)
- https://huggingface.co/meta-llama/Llama-3.2-1B
- https://huggingface.co/allenai/OLMo-2-0425-1B
- https://huggingface.co/HuggingFaceTB/SmolLM3-3B
- https://huggingface.co/datasets/ai4bharat/sangraha

Papers + frameworks:
- WSM: https://openreview.net/forum?id=HhThhjKyfw (arxiv 2507.17634)
- muP: Yang et al., Tensor Programs V (arxiv 2203.03466)
- EU AI Act: https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai

Project artifacts:
- First external review: `docs/MyLLM_Repo_Technical_Review_2026-05-12.docx` (committed)
- Second external review: PDF in handoff conversation; this document is the markdown summary
