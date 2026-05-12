# MyLLM v1 — Model Card
*Status: TEMPLATE / SCAFFOLDING (B9, 2026-05-12 audit). Filled in per actual run state at v1 release.*

## Model details

- **Model name**: MyLLM-1B-Base-v1 (internal designator)
- **Developed by**: Samatva (lead: harshit.hv@samatva.com)
- **Model type**: Dense decoder-only Transformer (Llama-style)
- **Languages**: English (primary); Hindi (secondary); Spanish, Chinese, Arabic, French, German (tertiary)
- **License**: TBD (Apache-2.0 target)
- **Model versions**:
  - v1: 1T-token internal pretrain checkpoint (this card)
  - v1.5 (planned): continued to 3T tokens if v1 eval clears continuation criteria
- **Architecture**: 16 layers × 2048 hidden × 4× FFN × GQA 16:4 × tied embeddings × RMSNorm + SwiGLU + RoPE (base 130K) + QK-norm
- **Training tokens**: 1T (target for v1)
- **Tokenizer**: SentencePiece Unigram, 131k vocab, NFKC + Metaspace, byte_fallback
- **Context length**: 4k (base) — long-context anneal to 32k via YaRN in v1.x (Phase 4 follow-up)

## Intended use

- Internal research model + capability validation for the MyLLM training stack
- NOT intended for production deployment in v1 form
- Post-trained variants (SFT/DPO) targeted as the user-facing product

## Out-of-scope use

- High-stakes decisions (medical, legal, financial advice)
- Generating content for jurisdictions where the model's training data wasn't licensed for derivative use
- Real-time safety-critical applications

## Training data

See [data_card_v1.md](data_card_v1.md). Summary:
- 44% FineWeb-Edu (HuggingFaceFW/fineweb-edu, ODC-By)
- 18% bigcode/the-stack-v2 (BigCode T&Cs accepted)
- 13.5% nvidia/Nemotron-CC (gated — pending NVIDIA approval; placeholder)
- 6% Wikipedia (wikimedia/wikipedia, CC-BY-SA)
- 5% pg19 (public domain)
- 6% allenai/peS2o (ODC-By)
- 7% open-web-math/open-web-math (ODC-By)
- 2% HuggingFaceH4/stack-exchange-preferences (CC-BY-SA)
- 4% ai4bharat/sangraha (CC-BY-4.0) — Hindi
- 8% mc4 → allenai/c4 multilingual (es/zh/ar/fr/de) (ODC-By)

## Training methodology

- WSD (Warmup-Stable-Decay) LR schedule
- muP per-parameter LR scaling, base_width=256
- WSM (Warmup-Stable-Merge): merge last 3-5 stable-phase checkpoints
- Decay-phase logit distillation from **DeepSeek-V4-Pro-Base** (MIT) and **Olmo-3-1125-32B** (Apache-2.0). α annealed 0.7 → 0.3 across the decay window. Top-K=8 cached logits per token. See [`teacher_distillation_strategy.md`](../teacher_distillation_strategy.md).
- Intra-document attention masking via `segment_ids`
- Atomic NaN-skip in `train_step` (no batch drift)
- Pre-launch contamination index over MMLU-ProX + Belebele + MILU prompts

## Evaluation

Held-out gates planned for v1:
- **MMLU-Pro** — general reasoning
- **MMLU-ProX** — multilingual MMLU across 29 languages
- **Belebele** — 122-language reading comprehension
- **MILU** — 11-language Indic understanding
- **HumanEval+** / **MBPP+** — code
- **LiveCodeBench** — contamination-free coding
- **GSM8K** / **MATH** / **MGSM** — math
- **RULER** — long-context (replaces NIAH)
- **Calibration**: ECE per prompt category
- **Toxicity / bias**: multilingual safety suites
- **Memorization probes** — verbatim string regurgitation tests

(Phase C / Phase 3 will fill in actual numbers per benchmark.)

## Limitations

- Trained on 1T tokens — below frontier 1B-class budgets (Llama 3.2 1B: 9T; SmolLM3 3B: 11.2T; OLMo 2 1B: 4T). v1 is positioned as "internal validation"; continuation to 3T planned.
- English-primary. Hindi at 4% of pretrain is enough for basic reading, may be weaker on generation.
- No code data from gated bigcode/the-stack-v2 in the v1 cycle until T&C signoff finalizes. Substituted with starcoderdata or downscale code share if not.
- Distillation from base teachers only (no chat/instruct); model has no inherent instruction-following capability — that lives in the post-trained variants.

## Environmental impact

(Filled per actual run state.)
- Hardware: TBD (target 8× H200 SXM secure cloud, RunPod)
- Training time: TBD (target ~24 days for 1T tokens)
- Carbon: TBD (compute provider's mix; documented in v1 release notes)

## Citation

(Filled at v1 release.)

## References

- Project handoff: [project_handoff_2026-05-11.md](../project_handoff_2026-05-11.md)
- Teacher strategy: [teacher_distillation_strategy.md](../teacher_distillation_strategy.md)
- External technical reviews: `docs/MyLLM_Repo_Technical_Review_2026-05-12.docx` + `docs/external_review_2026-05-12_enterprise.md`
