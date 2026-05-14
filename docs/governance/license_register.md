# License Register - MyLLM Base Program

Status: live draft, refreshed for the pre-2 plan on 2026-05-14.

This register records dataset licenses, model-license constraints, excluded
sources, benchmark licenses, and release-posture assumptions for the MyLLM
1B-class base-model program. It pairs with `data_card_v1.md`,
`model_card_v1.md`, and the current plan in
`docs/PRE2_FINAL_PLAN_2026-05-14.md`.

## Pretraining Dataset Candidates

| Source | License / terms posture | Current use | Notes |
|---|---|---|---|
| HuggingFaceFW/fineweb-edu | ODC-By 1.0 | Candidate / likely core source | Requires attribution tracking and quality-slice accounting. |
| bigcode/the-stack-v2 | BigCode Open RAIL-M v1 | Candidate code source | T&C accepted 2026-05-11; downstream use restrictions must be mirrored in release docs. |
| Wikimedia / Wikipedia | CC-BY-SA 4.0 | Candidate, capped | Share-alike/legal interpretation needs counsel before public Apache-2.0 weight release. |
| Project Gutenberg / pg19-like public-domain books | Public domain where verified | Candidate, capped | Use only verified public-domain material. |
| allenai/peS2o | ODC-By 1.0 | Candidate science source | Track source provenance. |
| open-web-math/open-web-math | ODC-By 1.0 | Candidate math source | Preferred over fragile Proof-Pile-2 ingestion. |
| Stack Exchange-derived data | CC-BY-SA 4.0 / site terms | Candidate, capped | Share-alike and attribution implications need legal review. |
| ai4bharat/sangraha | CC-BY-4.0 | Candidate Indic source | Useful for Hindi/Indic coverage; verify exact subset licenses. |
| C4 / multilingual C4-like data | ODC-By 1.0 where using AllenAI C4 | Candidate multilingual source | Keep language splits explicit. |
| Dolma / OLMo-family sources | Mixed, source-specific | Candidate after source-level license review | Do not import as a black box; register each constituent source. |
| DCLM / FineWeb-style high-quality web data | License varies by artifact | Candidate after source-level license review | Use quality filters, dedup, and source accounting. |

## Excluded Or Blocked Sources

| Source | Status | Reason |
|---|---|---|
| nvidia/Nemotron-CC | Blocked unless access and terms are approved | Gated source; no training use until access and license review are complete. |
| EleutherAI/proof-pile-2 | Excluded for current pipeline | Loader fragility and decompression failures; math coverage should come from more stable sources. |
| Unlicensed crawls, scraped social/private content, leaked corpora | Excluded | Fails release, governance, and reproducibility standards. |
| API-generated outputs from closed commercial LLMs | Excluded | Typical terms restrict using outputs to develop competing models. |

## Teacher And Distillation Policy

The current pre-2 decision is to train the base model primarily from real
corpus data. Heterogeneous-tokenizer top-K logit distillation is not part of
the foundation training plan. Distillation may return later only as a
bounded experiment with permissive teachers, tokenizer-aligned targets, legal
review, and clear ablation wins.

| Teacher family | Current status | Reason |
|---|---|---|
| Open permissive base models such as OLMo-family checkpoints | Possible future experiment | Need exact checkpoint license, tokenizer compatibility, and quality-gate evidence. |
| DeepSeek-family checkpoints | Not locked | Requires exact current checkpoint and license review before use. |
| Llama-family checkpoints | Excluded for derivative-weight release unless counsel approves | Community license terms impose naming and large-user restrictions. |
| Gemma-family checkpoints | Excluded for training/distillation into this base model | Model terms can cover derivatives and synthetic outputs. |
| GPT / Claude / Gemini API outputs | Excluded | Closed-provider terms commonly prohibit training competing models on outputs. |
| Mistral checkpoints with revenue or derivative restrictions | Excluded | Restrictive clauses are not acceptable for a clean foundation release. |

## Planned Output License

Target posture: Apache-2.0 for code and, if legal review clears the final data
mix, Apache-2.0 or another permissive license for weights.

Open legal risks before any public release:

1. CC-BY-SA sources may impose attribution or share-alike obligations.
2. RAIL-style code-data restrictions may need to be propagated into acceptable-use terms.
3. Every gated dataset must have auditable terms acceptance and redistribution review.
4. The data card must report real token shares from the final manifest, not planning estimates.

## Benchmark Datasets

Benchmark text may be ingested for decontamination indexes and evaluation
prompts, not for training. Register exact versions before release.

| Benchmark | License posture | Use |
|---|---|---|
| MMLU-Pro / MMLU-ProX | Permissive or dataset-specific; verify exact source | Eval + decontamination |
| Belebele | CC-BY-SA 4.0 | Eval + decontamination |
| MILU | CC-BY 4.0 | Eval + decontamination |
| HumanEval+ / MBPP+ | MIT-style; verify exact source | Code eval + decontamination |
| GSM8K / MATH / MGSM | Verify exact dataset license and version | Math eval + decontamination |
| BBH / IFEval | Apache-2.0 or source-specific; verify exact version | Reasoning / instruction eval |
| LiveCodeBench | Versioned benchmark | Eval only; decontam by release window, not static old snapshot |
| RULER | Synthetic eval | Generated at eval time |

## Acceptance Log

| Date | Action | By |
|---|---|---|
| 2026-05-11 | bigcode/the-stack-v2 T&C accepted on Hugging Face | harshit.hv@samatva.com |
| 2026-05-11 | nvidia/Nemotron-CC access requested | harshit.hv@samatva.com |
| 2026-05-14 | Old locked-teacher plan removed from current governance docs | Codex |
| 2026-05-14 | Pre-2 data, teacher, and license posture aligned to current plan | Codex |

## Update Policy

Any dataset, teacher, benchmark, or release-license change must update this
register in the same PR as the config or manifest change. Preserve excluded
entries for audit history, but avoid linking to deleted review drafts.
