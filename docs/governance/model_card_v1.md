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
<!-- AUTO:start name="architecture" -->
- **Architecture**: 16 layers × 2048 hidden × 4× FFN × GQA 32:8 × tied embeddings × RMSNorm + SwiGLU + RoPE (base 500,000) + QK-norm
- **Training tokens**: 1T (target for v1)
- **Tokenizer**: SentencePiece Unigram, 131k vocab, NFKC + Metaspace, byte_fallback
- **Context length**: 8k natively — long-context anneal to 32k via YaRN
- **Mixed precision**: bfloat16
<!-- AUTO:end -->

> **Note**: the architecture block above is auto-rendered from `configs/base_1b.yaml` (and the sources table from `configs/data/pretrain_mix.yaml`) via `scripts/render_governance_cards.py`. Regenerate at every release to avoid template drift. The values below are accurate as of 2026-05-12 commit.

## Intended use

- Internal research model + capability validation for the MyLLM training stack
- NOT intended for production deployment in v1 form
- Post-trained variants (SFT/DPO) targeted as the user-facing product

## Out-of-scope use

- High-stakes decisions (medical, legal, financial advice)
- Generating content for jurisdictions where the model's training data wasn't licensed for derivative use
- Real-time safety-critical applications

## Training data

See [data_card_v1_template.md](data_card_v1_template.md). Summary (matches the LIVE configs/data/pretrain_mix.yaml as of 2026-05-12; Nemotron-CC excluded pending NVIDIA approval, its share absorbed by FineWeb-Edu):
- **44%** FineWeb-Edu (HuggingFaceFW/fineweb-edu, ODC-By)
- **18%** bigcode/the-stack-v2 (BigCode T&Cs accepted)
- **6%** Wikipedia (wikimedia/wikipedia, CC-BY-SA, config 20231101.en)
- **5%** pg19 (public domain)
- **6%** allenai/peS2o (ODC-By)
- **7%** open-web-math/open-web-math (ODC-By; absorbs proof-pile-2's math share since proof-pile-2 was dropped permanently after loader fragility)
- **2%** HuggingFaceH4/stack-exchange-preferences (CC-BY-SA)
- **4%** ai4bharat/sangraha split=hin (CC-BY-4.0) — Hindi only
- **8%** mc4 → allenai/c4 multilingual (es:1.5% / zh:1.5% / ar:1.5% / fr:1.5% / de:2%) (ODC-By)

Total: 100%.

Excluded from this v1 mix:
- **nvidia/Nemotron-CC** (was 13.5%): gated, NVIDIA approval pending. Re-include when access lands. Absorbed by FineWeb-Edu.
- **EleutherAI/proof-pile-2**: dropped permanently after multiple loader failures (zstd decompression error mid-stream).

## Training methodology

- WSD (Warmup-Stable-Decay) LR schedule
- muP per-parameter LR scaling, base_width=256
- WSM (Warmup-Stable-Merge): merge last 3-5 stable-phase checkpoints
- Decay-phase logit distillation from **DeepSeek-V4-Pro-Base** (MIT) and **Olmo-3-1125-32B** (Apache-2.0). α annealed 0.7 → 0.3 across the decay window. Top-K=8 cached logits per token. See [`teacher_distillation_strategy.md`](../teacher_distillation_strategy.md).
- Intra-document attention masking via `segment_ids`
- Atomic NaN-skip in `train_step` (no batch drift)
- Pre-launch n-gram decontamination index covering: MMLU-ProX, Belebele, MILU, MMLU-Pro, HumanEval+, MBPP+, GSM8K, MATH, MGSM, BBH, IFEval (13-grams, xxhash64; see `scripts/build_decontamination_index.py`)

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

- Live session handoff: [../SESSION_HANDOFF.md](../SESSION_HANDOFF.md)
- Design reference: [../DESIGN.md](../DESIGN.md)
- Teacher strategy: [../teacher_distillation_strategy.md](../teacher_distillation_strategy.md)
- External technical reviews (archived): `docs/archive/MyLLM_Repo_Technical_Review_2026-05-12.docx` + `docs/archive/external_review_2026-05-12_enterprise.md`
