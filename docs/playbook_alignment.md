# Playbook Alignment — v1 (2026-05-10)

Decision log mapping the "Building Bharat's Foundation: A $30M Sovereign LLM Playbook"
recommendations against the MyLLM project. Strategic position chosen by the
project owner: **Path B (generic English-primary small LLM) with selective
sovereign hedges.**

## Strategic position

We are NOT positioning as a sovereign Indian foundation model. We ARE
adopting universal best-practice improvements from the playbook plus a few
low-cost sovereignty hedges that preserve optionality without committing to
a sovereign-Indian product narrative.

## Adopted (universal improvements — applied this commit)

| # | Change | Reason |
|---|---|---|
| U1 | **Vocab 128000 → 131072** | Llama-3.x compatibility; trivial ~2.4M-param bump |
| U2 | **Tokenizer: byte-level BPE → SentencePiece-Unigram** | Better cross-lingual quality; standard for modern LMs (Sarvam, Llama 1/2, Mistral); byte-fallback handles arbitrary scripts |
| U3 | **NFKC normalization** | Canonical Unicode form; eliminates Unicode-encoding ambiguity |
| U4 | **Metaspace pre-tokenizer** | SentencePiece-style; handles whitespace consistently across scripts |
| U5 | **WSD (Warmup-Stable-Decay) schedule** | Doesn't commit to token budget upfront; can anneal any stable-phase checkpoint; SmolLM2 + MiniCPM standard |
| U6 | **Document teacher licensing constraint** | DeepSeek V3.2 (MIT) is the only "premium teacher" with no naming/usage clause; Gemma forbids using outputs to train competing models; Llama requires "Llama-" naming on derivatives |
| U7 | **`<\|unk\|>` token added** | Required by Unigram model; placeholder when byte-fallback isn't applicable |

## Adopted (sovereign hedges — applied this commit)

| # | Change | Reason |
|---|---|---|
| S1 | **Bump multilingual share 8% → 12%, source Hindi from AI4Bharat Sangraha** | Higher-quality Hindi than mc4; Sangraha is CC-BY-4.0 (commercial-friendly) |
| S2 | **Document IndiaAI Mission as compute path** | 40-47% subsidy on portal rates ($1.20/GPU-hr vs $5/hr cloud) — cuts 1B base run from ~$300k to ~$80k. Worth applying for. |
| S3 | **Bharat-* evals listed as supplementary** | Bharat-Law-Bench, Bharat-Civics, Bharat-Code-Mix as private rotating evals if/when we want a sovereign-pivot escape hatch |
| S4 | **DPDP Act posture in plan** | Don't store user PII without consent; relevant if we ever serve Indian users |

## Deliberately NOT adopting (Path A only)

| # | Recommendation | Why we're skipping |
|---|---|---|
| N1 | Vocab 256k (full 22-Indic-language coverage) | Embedding cost (262M extra params at 2048 hidden); we only need 7 languages |
| N2 | 30-40% Indic data mix | Inappropriate for English-primary product; current 12% multilingual is sufficient |
| N3 | Indic grapheme pre-segmentation | Real implementation cost; defer to v2 if we pivot toward Indic depth |
| N4 | License Indian publishers (HT, ABP, Sun, Sakal, Eenadu, Mathrubhumi) | $1M+ in licensing fees; only justified for sovereign positioning |
| N5 | Sovereign Indian branding | Strategic decision belongs to product, not engineering |
| N6 | Apply for IndiaAI Mission Foundation-Model pillar grant (~$30M) | Selection process is competitive; not the right ask for a non-sovereign product |
| N7 | QK-norm default-on at 1B scale | Llama 3.2 1B doesn't use it; defer until pilot loss curves justify it |
| N8 | MoE for 7B+ | Playbook agrees this is a trap below 30B; aligned with our plan to stay dense |

## Validated by playbook (no change needed — already aligned)

| Recommendation | Our state |
|---|---|
| Sovereign-1B = 16L/2048H/32-8/64dim/FFN8192/tied | exact match in `base_1b.yaml` |
| Pre-norm decoder, RMSNorm, SwiGLU, GQA | match in `layers.py` |
| RoPE base 500,000 | match (after architecture review) |
| AdamW β=0.9/0.95, wd=0.1, grad-clip 1.0 | match |
| BF16 weights + FP8 mixed precision | planned |
| `<\|tool_call\|>{json}<\|/tool_call\|>` format | match in `SpecialTokens` |
| vLLM/SGLang/llama.cpp serving stack | match in PLAN.md |
| No Llama derivatives in any path | enforced |
| Apache-2.0 release | enforced |
| Watchdog auto-rollback + checkpoint cadence | wired this turn |
| Distillation > RL at small scale; GRPO if RL | documented in `math_strategy.md` |
| Refuse >25% revenue from related party | governance, no engineering action |

## Open items pending decision

- **Teacher API model identity** — user has not confirmed which model their "chat API" wraps. If it's Llama-derived, we cannot use its outputs without Llama-naming compliance. If Gemma-derived, we cannot use its outputs at all. **Action: confirm before Phase 6 (SFT).**
- **IndiaAI Mission application** — would cut training cost ~3-5×. Application process and timeline TBD. **Action: scope this as a Phase 0 task if user wants to pursue.**
- **Bharat-* eval set construction** — listed as supplementary; if we want them, scope is ~2 weeks of data engineering. **Action: defer until we know whether we want a sovereign-pivot option.**
