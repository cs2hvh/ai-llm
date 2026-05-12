# License Register — MyLLM v1
*Status: SCAFFOLDING (B9, 2026-05-12 audit). Filled per actual run state at v1 release.*

This document is the authoritative record of every dataset license, model
license, and license clause that constrains the MyLLM v1 model's training,
distillation, and release. It pairs with the data card and model card.

Required by:
- EU AI Act GPAI obligations (Aug 2, 2025): copyright policy + transparency
- ISO/IEC 42001 lifecycle process
- NIST AI RMF GenAI profile

---

## Pretraining datasets

| HF dataset | License | License text URL | T&C accepted? | Notes |
|---|---|---|---|---|
| HuggingFaceFW/fineweb-edu | ODC-By 1.0 | https://opendatacommons.org/licenses/by/1-0/ | N/A (open) | |
| bigcode/the-stack-v2 | BigCode Open RAIL-M v1 | https://huggingface.co/datasets/bigcode/the-stack-v2 | ✅ accepted 2026-05-11 | Use-restriction clauses re weapons / hate speech / etc. |
| wikimedia/wikipedia | CC-BY-SA 4.0 | https://creativecommons.org/licenses/by-sa/4.0/ | N/A | Share-alike: derivatives must be CC-BY-SA |
| pg19 | Public domain (Project Gutenberg pre-1919) | https://www.gutenberg.org/policy/license.html | N/A | |
| allenai/peS2o | ODC-By 1.0 | https://huggingface.co/datasets/allenai/peS2o | N/A | |
| open-web-math/open-web-math | ODC-By 1.0 | https://huggingface.co/datasets/open-web-math/open-web-math | N/A | |
| HuggingFaceH4/stack-exchange-preferences | CC-BY-SA 4.0 | https://stackexchange.com/legal | N/A | |
| ai4bharat/sangraha | CC-BY-4.0 | https://huggingface.co/datasets/ai4bharat/sangraha | N/A | 251B tokens / 22 langs; Hindi = 34.5B (verified 2026-05-12) |
| mc4 (deprecated → allenai/c4) | ODC-By 1.0 | https://huggingface.co/datasets/allenai/c4 | N/A | |

### Excluded pretrain sources (with reason)

| HF dataset | Status | Why |
|---|---|---|
| nvidia/Nemotron-CC | EXCLUDED | NVIDIA gating; manual approval pending (as of 2026-05-12). Re-include if/when access lands. |
| EleutherAI/proof-pile-2 | EXCLUDED PERMANENTLY | Loader fragility (zstd decompression failures mid-stream); 4 separate failure modes during pre-launch smoke. Math slice absorbed by open-web-math. |

---

## Distillation teachers

Locked teacher plan v2 — see `docs/teacher_distillation_strategy.md` for full reasoning.

| Teacher model | License | Critical clauses | Allowed for our use? |
|---|---|---|---|
| `deepseek-ai/DeepSeek-V4-Pro-Base` | MIT | None — fully permissive | ✅ YES — locked as primary teacher |
| `allenai/Olmo-3-1125-32B` | Apache-2.0 | None — fully permissive | ✅ YES — locked as secondary teacher (production phase only, after canary) |

### Teachers we EXPLICITLY excluded (and why)

| Teacher | License | Critical clause | Verdict |
|---|---|---|---|
| `mistralai/Mistral-Medium-3.5-128B` | "Modified MIT" | *"You are not authorized to exercise any rights under this license if the global consolidated monthly revenue of your company exceeds $20 million."* — applies to derivatives + combined works | EXCLUDED. Distilled logits would arguably create a derivative; the $20M cap is a perpetual time-bomb on every weight that comes from a run including this teacher. |
| `Qwen/Qwen3.6-27B` | Apache-2.0 | (technical, not legal): "Type: Causal Language Model with Vision Encoder" + thinking-mode-default | EXCLUDED. Not a base text-only model. Distilling would bake `<think>` traces + vision-encoder coupling into our base. |
| Any Llama 3.x / Llama 4 | Llama Community License | 700M-MAU clause + naming-prefix requirement on derivatives | EXCLUDED. Same class of restriction as Mistral's revenue clause. |
| Gemma 2 / 3 / 4 (Google) | Gemma TOS | §3.2 "Model Derivatives" explicitly covers "synthetic data Outputs by Gemma" | EXCLUDED. Forbidden for distillation. |
| GPT-4 / Claude / Gemini API outputs | Various ToS | "do not use outputs to develop competing models" | EXCLUDED. |
| `deepseek-ai/DeepSeek-V3-Base` | DeepSeek Model License (custom, non-OSI) | Custom commercial-use clauses; legally requires per-use review | EXCLUDED for v1 (redundant with V4-Pro family anyway). |

---

## Output license posture

**Planned**: MyLLM-1B-Base-v1 weights released under **Apache-2.0**.

This is feasible because:
- All training-data licenses (ODC-By, CC-BY-SA, CC-BY-4.0, public domain) permit derivative use
- All distillation-teacher licenses (MIT, Apache-2.0) permit derivative use without naming or revenue restrictions
- All data filters (length, repetition, PII redaction) are own-implementation
- The model code is our own (Apache-2.0 friendly)

**Caveats / open questions** (legal review before release):
- CC-BY-SA share-alike: technically a model trained on Wikipedia-style data may need to honor share-alike on the model's outputs. Industry practice has been to argue training is fair use / not subject to share-alike. Need counsel sign-off before Apache-2.0 release if Wikipedia stays at >5% of pretrain mix.
- bigcode/the-stack-v2 RAIL-M clauses: the use-restrictions section is binding on the distributor. Our model card must include the same use-restriction clauses (or successor language). To be drafted.

---

## Benchmark datasets (decontamination + eval)

The pretrain corpus is n-gram-decontaminated against these benchmarks. The
*benchmark* license matters here because we ingest prompt text into the
decontamination index (which lives alongside training artifacts) — not for
training, but for filtering. All eight extended-gate benchmarks below are
permissive (MIT or Apache-2.0).

| Benchmark id | HF dataset | License | Used for |
|---|---|---|---|
| mmlu-prox | li-lab/MMLU-ProX | MIT | Decon + eval |
| belebele | facebook/belebele | CC-BY-SA 4.0 | Decon + eval |
| milu | ai4bharat/MILU | CC-BY 4.0 | Decon + eval |
| mmlu-pro | TIGER-Lab/MMLU-Pro | MIT | Decon + planned eval |
| humaneval-plus | evalplus/humanevalplus | MIT | Decon + planned eval |
| mbpp-plus | evalplus/mbppplus | MIT | Decon + planned eval |
| gsm8k | openai/gsm8k | MIT | Decon + planned eval |
| math | HuggingFaceH4/MATH-500 | MIT | Decon + planned eval |
| mgsm | juletxara/mgsm | MIT | Decon + planned eval |
| bbh | maveriq/bigbenchhard | Apache-2.0 | Decon + planned eval |
| ifeval | google/IFEval | Apache-2.0 | Decon + planned eval |

LiveCodeBench and RULER are intentionally NOT in the decontamination
index: LiveCodeBench is release-versioned (re-index per release in Phase C
when eval lands), RULER is synthetic and generated at eval time.

---

## Acceptance log

| Date | Action | By |
|---|---|---|
| 2026-05-11 | bigcode/the-stack-v2 T&Cs accepted on HuggingFace | harshit.hv@samatva.com |
| 2026-05-11 | nvidia/Nemotron-CC access requested | harshit.hv@samatva.com (PENDING) |
| 2026-05-12 | Mistral-Medium-3.5-128B excluded after license file ($20M cap) verification | harshit.hv@samatva.com |
| 2026-05-12 | Qwen3.6-27B excluded after modality verification (multimodal + thinking) | harshit.hv@samatva.com |
| 2026-05-12 | DeepSeek-V4-Pro-Base locked as primary teacher | harshit.hv@samatva.com |
| 2026-05-12 | Olmo-3-1125-32B locked as secondary teacher (production-only) | harshit.hv@samatva.com |

---

## Update policy

Any new dataset or teacher addition MUST:
1. Update this register with license + URL
2. Get T&C acceptance if gated
3. Be reflected in `configs/data/pretrain_mix.yaml` AND `configs/decay_phase_distillation.yaml`
4. Update `docs/data_card_v1_template.md`
5. Be verified via dossier WebFetch (per `feedback_verify_before_locking.md`)

Removals follow the same process — preserve the entry in the "EXCLUDED" sections for audit history.
