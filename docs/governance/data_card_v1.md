# MyLLM v1 — Data Card
*Status: TEMPLATE / SCAFFOLDING (B9, 2026-05-12 audit). Filled in per actual run state at v1 release.*

This data card documents the pretraining corpus per EU AI Act GPAI
obligation (effective Aug 2, 2025): "**a publicly available, sufficiently
detailed summary of the content used for training**."

## Pretraining corpus summary

**Total tokens trained on (v1 target)**: ~1 trillion
**Tokenizer**: SentencePiece Unigram, 131k vocab (`artifacts/tokenizer_v1.json`)
**Decontamination**: 13-gram n-gram overlap against MMLU-ProX + Belebele + MILU benchmark prompts (per OLMo-2 convention)
**License manifest**: see [license_register.md](license_register.md)

### Sources

<!-- AUTO:start name="sources" -->
| # | HF dataset | Configured share | License | Revision pinned? | Notes |
|---|---|---|---|---|---|
| 1 | HuggingFaceFW/fineweb-edu | 44.0% | ODC-By 1.0 | ⏳ B2 work | High-quality educational subset; absorbs Nemotron-CC share pending NVIDIA approval |
| 2 | bigcode/the-stack-v2 | 18.0% | BigCode Open RAIL-M (T&Cs accepted 2026-05-11) | ⏳ B2 work | Code |
| 3 | wikimedia/wikipedia (20231101.en) | 6.0% | CC-BY-SA 4.0 | ⏳ B2 work | English Wikipedia snapshot |
| 4 | pg19 | 5.0% | Public domain (Project Gutenberg pre-1919) | ⏳ B2 work | Books |
| 5 | allenai/peS2o | 6.0% | ODC-By 1.0 | ⏳ B2 work | Academic |
| 6 | open-web-math/open-web-math | 7.0% | ODC-By 1.0 | ⏳ B2 work | Math; absorbs proof-pile-2 share (dropped due to loader fragility) |
| 7 | HuggingFaceH4/stack-exchange-preferences | 2.0% | CC-BY-SA 4.0 | ⏳ B2 work | Q&A (question field only) |
| 8 | ai4bharat/sangraha (verified, split=hin) | 4.0% | CC-BY-4.0 | ⏳ B2 work | Hindi sovereign hedge |
| 9 | mc4 (es) | 1.5% | ODC-By 1.0 | ⏳ B2 work | Secondary language |
| 10 | mc4 (zh) | 1.5% | ODC-By 1.0 | ⏳ B2 work | Secondary language |
| 11 | mc4 (ar) | 1.5% | ODC-By 1.0 | ⏳ B2 work | Secondary language |
| 12 | mc4 (fr) | 1.5% | ODC-By 1.0 | ⏳ B2 work | Secondary language |
| 13 | mc4 (de) | 2.0% | ODC-By 1.0 | ⏳ B2 work | Secondary language |

**Total**: 100.0% (validated at run-start; mixture sampler is token-weighted per P0-6 fix)
<!-- AUTO:end -->

### Permanently excluded from training

| Source | Why excluded |
|---|---|
| nvidia/Nemotron-CC | Gated; NVIDIA approval pending. Was originally 13.5% of v1 plan; FineWeb-Edu absorbed the share. Re-include when access lands. |
| EleutherAI/proof-pile-2 | Loader-script fragility (4 separate failure modes in pre-launch smoke). Math share absorbed by open-web-math. |
| Any Llama/Gemma model output | License clauses make derivative use legally fraught for our distillation pipeline. |
| Any OpenAI/Anthropic/Google API output | ToS forbid using outputs to develop competing models. |

### Filters applied per document

- **Length**: 200 ≤ chars ≤ 1,000,000
- **Repetition**: top-word share ≤ 20%; top-5gram share ≤ 10%
- **Symbol ratio**: ≤ 30% non-alphanumeric
- **PII redaction**: email + phone (IPv4 NOT redacted)
- **Decontamination**: drop any document containing a 13-gram present in MMLU-ProX, Belebele, or MILU prompts

### Filters NOT applied (gaps to document)

- ⏳ No NSFW classifier on pretrain corpus (v1 deferred to post-training safety pass)
- ⏳ No copyright-flag screen beyond source license inheritance
- ⏳ No verifiable-author screen on code data (the-stack-v2 already has opt-out via GitHub but not author authentication)
- ⏳ Minimal duplicate detection: per-doc MinHash only; no cross-document near-dedup yet (Phase B2 work)

## Per-source token counts (filled at run time)

(Run-time aggregated counts from the loop's `emitted_per_source` field on `MixtureSampler`. Phase B work to persist into the data card automatically.)

| Source | Tokens emitted | % of total trained on |
|---|---|---|
| FineWeb-Edu | TBD | |
| the-stack-v2 | TBD | |
| ... | | |

## Personal data handling (DPDP / GDPR / EU AI Act)

The training corpus may contain personal data published openly (e.g., names in Wikipedia articles, Stack Exchange usernames). We do NOT:
- Use any private / non-public data sources
- Cross-reference public mentions of individuals
- Re-publish PII verbatim — the model may regurgitate training examples in some cases; this is documented as a limitation (memorization probe gate at v1 release)

For DPDP / GDPR Article 21 (right to object) requests, contact harshit.hv@samatva.com. Removal from training data requires re-training; we maintain the manifest + revision pins to make this auditable.

## Audit trail

This data card pairs with:
- `configs/data/pretrain_mix.yaml` — the live mix configuration
- `license_register.md` — full license text + accepted T&Cs per source
- `quarantine.jsonl` (per-run, in `<checkpoint_root>/`) — every bad batch that the NaN-skip patch dropped, for forensic
- `decontamination_report.csv` — per-source contamination overlap stats
- The data manifest (`scripts/build_data_manifest.py`, B2 work) — per-shard SHA256 + source revision per packed sequence
