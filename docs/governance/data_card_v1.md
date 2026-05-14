# MyLLM 1B-Class Data Card

Status: live draft, refreshed for the pre-2 plan on 2026-05-14.

This data card is a release scaffold for the training-corpus disclosure. It
must be regenerated from the final data manifest before any checkpoint is
published.

## Corpus Target

- Training target: at least 1T tokens, with a pre-2 planning target of 1.5T tokens.
- Tokenizer: TBD for pre-2; do not reuse pre-1 tokenizer assumptions without a fresh tokenizer study.
- Corpus shape: deduplicated, decontaminated, source-accounted, license-reviewed, and version-pinned.
- Manifest requirement: every emitted shard must carry source id, source revision, filter version, dedup version, tokenizer version, token count, and content hash.

## Candidate Source Families

| Family | Purpose | Current status | Required controls |
|---|---|---|---|
| Educational / high-quality web | Broad language modeling and factual coverage | Candidate core source | Quality classifier, exact source license, dedup, PII filters |
| Curated general web | Breadth and robustness | Candidate, capped | Strong filtering, domain/source accounting, contamination checks |
| Code | Programming capability | Candidate, license-sensitive | Repository license review, benchmark decontam, generated-code contamination checks |
| Math and STEM | Reasoning and symbolic competence | Candidate | Stable loaders, source attribution, benchmark decontam |
| Books and reference | Long-form prose and knowledge | Candidate, capped | Public-domain or licensed-only material |
| Q&A / documentation | Explanatory style and factual density | Candidate, capped | Attribution/share-alike review and duplication checks |
| Indic / multilingual data | Hindi and broader multilingual coverage | Candidate strategic source | Language-id, script checks, source-level license review |
| Synthetic data | Targeted repair only | Bounded, tagged, post-core | Keep separate from organic corpus; require eval-backed inclusion |

## Excluded Data

| Data type | Reason |
|---|---|
| Private or non-public data | Fails privacy, governance, and reproducibility standards. |
| Leaked, pirated, or unclear-copyright corpora | Not acceptable for a release-quality foundation model. |
| Closed-model API outputs | Typical provider terms prohibit use for training competing models. |
| Unreviewed gated datasets | No use until access, terms, and redistribution constraints are approved. |
| Benchmark answer sets in training data | Contamination risk; benchmark text may be used only for decontamination/eval. |

## Processing Requirements

Every source must pass:

1. License and terms review recorded in `license_register.md`.
2. Source revision pinning.
3. Language identification where multilingual.
4. Document-level quality filtering.
5. PII filtering/redaction appropriate to the source.
6. Exact and near-duplicate removal.
7. Benchmark decontamination.
8. Token-count accounting by source family and language.
9. Holdout split creation before training.

## Minimum Release Tables

The final release data card must include:

| Required table | Filled from |
|---|---|
| Source family token counts | Final packed-corpus manifest |
| Dataset-level token counts and licenses | Source registry + manifest |
| Language distribution | Language-id pass over accepted documents |
| Filter rejection rates | Data build logs |
| Dedup removal rates | Dedup reports |
| Contamination removal rates | Decontamination reports |
| PII filter statistics | Privacy filter reports |
| Known gaps | Manual release review |

## Personal Data Handling

The corpus must not intentionally include private data. Public web sources may
still contain personal information, so release review must document:

- PII filter versions and measured hit rates.
- Known residual-risk categories.
- Contact path for removal or objection requests.
- Whether removal requires retraining, continued pretraining, or data exclusion in future runs.

## Audit Trail

This data card pairs with:

- `license_register.md`
- `model_card_v1.md`
- final data manifest
- decontamination reports
- dedup reports
- source registry
- tokenizer training report

Planning percentages are not release evidence. Use emitted token counts from
the final manifest for any public claim.
