# IndiaAI Mission — Compute Subsidy Application Brief
**Status: DRAFT v0.1 (2026-05-10)** — needs verification against the live
IndiaAI portal application form. Some fields are placeholders the project
owner must complete (marked `<TBD>`).

---

## 1. Project identity

| Field | Value |
|---|---|
| Project name | MyLLM — sovereign-hedged 1B-parameter foundation model |
| Lead applicant | Harshit HV (`harshit.hv@samatva.com`) |
| Legal entity | `<TBD: registered company / Section 8 / individual>` |
| GSTIN / CIN | `<TBD>` |
| Project category | Foundation model pretraining (decoder-only LLM) |
| Country of incorporation | India |
| Public/private | Private (commercial); Apache-2.0 release of weights + recipe |

## 2. Project summary (200 words)

MyLLM is a 1B-parameter, decoder-only transformer foundation model
trained from scratch on a curated open-source corpus. It is built in the
Llama-3.2 / SmolLM2 / Qwen-2.5 architectural class — pre-norm RMSNorm,
SwiGLU FFN, grouped-query attention with 4:1 KV ratio, RoPE positional
encoding (base 500,000) — with a bespoke 131,072-vocab SentencePiece
Unigram tokenizer covering English plus six secondary languages
(Hindi, Spanish, French, German, Chinese, Arabic).

The full lifecycle is in scope: tokenizer → 1T-token base pretrain →
math/code-heavy continued pretrain → SFT → DPO → reasoning → safety →
tool-use → eval → quantization → serving. All training scripts,
checkpoints, and evaluations will be released under Apache-2.0 with no
Llama- or Gemma-derived components, ensuring downstream users (academic
and commercial) face no naming or usage restrictions.

The model is designed for English-primary use with selective Indic
hedging — Hindi data is sourced from AI4Bharat Sangraha (CC-BY-4.0)
rather than the deprecated mc4 dataset, ensuring better-quality
India-context training data without committing the project to a
sovereign-Indian product narrative.

## 3. Compute ask (by phase)

Estimated using community-cloud H100 SXM rates of ~$2.79/GPU-hr on
international clouds; IndiaAI subsidy assumes the listed 40-47% reduction
for compute booked through approved Indian providers.

| Phase | Workload | GPU hours | List cost (USD) | At ~45% subsidy (USD) |
|---|---|---:|---:|---:|
| 1 | Tokenizer training | ~64 (CPU + 1× A40 8 hrs) | $250 | $137 |
| 2 | Pilot 250M (50B tokens) | ~180 (1× H100 SXM × 7.5 days) | $500 | $275 |
| 3 | Base 1B pretrain (1T tokens) | ~7,000 (8× H100 SXM × 36 days) | $25,000 | $13,750 |
| 4 | Continued pretrain (math/code, 80B tokens) | ~700 (8× H100 × 3.5 days) | $2,500 | $1,375 |
| 5 | SFT | ~500 (4× H100 × 5 days) | $1,800 | $990 |
| 6 | DPO | ~250 (4× H100 × 2.5 days) | $900 | $495 |
| 7-9 | Reasoning + safety + tool-use SFT | ~1,000 (4× H100 × 10 days) | $3,600 | $1,980 |
| 10 | Eval harness | ~150 (1× H100 × 6 days) | $400 | $220 |
| 11 | Quantization | ~30 (1× H100 × 1 day) | $80 | $44 |
| 12 | Serving canary | ~150 (1× L40S × 6 days) | $130 | $72 |
| | **Total** | **~10,000** | **~$35,000** | **~$19,000** |

**Subsidy ask: ~$16,000 over 12-15 calendar months.** All training
compute booked through IndiaAI-approved providers; we will provide
itemized invoices on each phase boundary.

## 4. Timeline

| Milestone | Target month |
|---|---|
| Tokenizer + Phase 0/1 complete | Month 1 |
| Pilot 250M loss curves stable | Month 3 |
| Base 1B pretrain start | Month 4 |
| Base 1B pretrain complete (1T tokens) | Month 7 |
| SFT + DPO + reasoning checkpoints | Month 9 |
| Eval + safety report public | Month 10 |
| Quantized serving canary | Month 11 |
| Public release (Apache-2.0) | Month 12 |

## 5. Public-good commitment

- **License**: Apache-2.0 for model weights, training scripts, evaluation
  harness, and the technical report. No commercial-use restriction. No
  attribution requirement beyond standard Apache-2.0 NOTICE.
- **Reproducibility**: full data mixture spec, hyperparameter file,
  random seeds, and exact tokenizer artifact published alongside weights.
- **Indian-language coverage**: Hindi quality at parity with Llama-3.2 1B
  on Indic perplexity benchmarks; lighter coverage for other Indic
  languages via byte-fallback (no script falls outside the tokenizer).
- **No Llama or Gemma derivatives**: training does not consume outputs
  from any model whose license restricts derivative training.
- **Safety**: refusal taxonomy (`docs/safety_policy.md`) covering
  CSAM, dangerous-weapon synthesis, mass casualty, and DPDP-aligned
  PII handling.

## 6. Risk register

| Risk | Mitigation |
|---|---|
| Capacity unavailability at IndiaAI providers during Phase 3 | Reserve 8× H100 cluster time 30 days in advance; international cloud as fallback (lose subsidy, project still completes) |
| Loss spike during 1T-token run | Watchdog auto-rollback wired (restore + halve LR + skip batches); proven in synthetic data smoke |
| Tokenizer quality regression on Indic | Validation gates: per-language compression floor 0.85 on held-out test set |
| Teacher-API license drift (SFT phase) | Locked: only DeepSeek-V3.2 (MIT) outputs are usable; Gemma forbidden, Llama-derived requires renaming |
| Reproducibility audit | All checkpoints + manifests committed to Cloudflare R2 (sovereign-storage compatible); SHA-256 published |

## 7. Team

| Role | Person | Affiliation |
|---|---|---|
| Lead | Harshit HV | `<TBD>` |
| Compute / infra | `<TBD>` | `<TBD>` |
| Data engineering | `<TBD>` | `<TBD>` |
| Eval + safety | `<TBD>` | `<TBD>` |

## 8. Annexes referenced

- `PLAN.md` — full 14-phase lifecycle plan with hard gates
- `docs/architecture_review.md` — architecture comparison vs Llama-3.2 1B / SmolLM2 / Qwen-2.5
- `docs/playbook_alignment.md` — strategic position statement (Path B + sovereign hedges)
- `docs/math_strategy.md` — math handling across the lifecycle
- `docs/safety_policy.md` — refusal taxonomy v0.1
- `configs/base_1b.yaml`, `configs/pilot_250m.yaml`, `configs/tokenizer.yaml` — locked specs

## 9. Application checklist (verify before submission)

- [ ] Confirm exact IndiaAI portal URL and current eligibility criteria
- [ ] Fill `<TBD>` fields above (legal entity, team, affiliations)
- [ ] Attach Annex documents from `docs/`
- [ ] Confirm subsidized provider list and current rates (rates drift quarterly)
- [ ] Check whether Foundation Model Pillar grant ($30M) is a separate
      stream we want to apply to (current decision per `playbook_alignment.md`
      §N6: NOT applying — selection process is competitive and we are not
      sovereign-positioned)
