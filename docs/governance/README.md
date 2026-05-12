# Governance artifacts

This directory holds the enterprise governance documents required for
external release per:

- **EU AI Act GPAI obligations** (effective 2025-08-02): technical
  documentation, copyright policy, public summary of training content
- **ISO/IEC 42001** AI management-system structure
- **NIST AI RMF GenAI Profile**
- **OWASP GenAI Security Project** + Agentic Top 10
- **India MeitY AI Governance Guidelines** (Nov 2025) + DPDP framework

## Files

| File | Purpose | Status |
|---|---|---|
| `model_card_v1_template.md` | Per-release model card scaffold (capabilities, limitations, eval, citation) | TEMPLATE — fill at v1 release |
| `data_card_v1_template.md` | Training corpus disclosure (EU AI Act transparency requirement) | TEMPLATE — auto-populate from `MixtureSampler.emitted_per_source` at release |
| `license_register.md` | Authoritative log of every dataset and teacher license + clauses + T&C acceptance dates | LIVE — update on every change |

## Filing cadence

| Artifact | When updated |
|---|---|
| `license_register.md` | Every time a dataset or teacher is added / removed / re-licensed. Update synchronously with the relevant yaml change. |
| `model_card_v1_template.md` → versioned `model_card_v1.md` | At each released checkpoint (v1, v1.5, v2). Fork the template, fill in actual run state. |
| `data_card_v1_template.md` → versioned `data_card_v1.md` | At each release. Pull token counts from `emitted_per_source`. |

## Phase 5 (governance) work that still needs to happen

The templates here are SCAFFOLDING — they capture the structure but most numbers/details get filled in per-release. The Phase 5 workstream covers:

1. ⏳ `eval_card_v1.md` — every benchmark result with provenance + version pin
2. ⏳ `risk_card_v1.md` — known failure modes + mitigations + remaining gaps
3. ⏳ `model_supply_chain.md` — signing keys + provenance per training run
4. ⏳ `incident_response.md` — what to do when a regression / safety failure surfaces post-release
5. ⏳ EU AI Act "technical documentation" packet (formal compilation of the above for regulators)
6. ⏳ India DPDP workflow doc (PII handling + deletion / retention policy)
7. ⏳ NIST AI RMF self-assessment mapping (which controls apply, where)
8. ⏳ OWASP Agentic Top 10 mitigation matrix (for the eventual tool-use SFT phase)

## External reviews driving this scaffolding

- `docs/MyLLM_Repo_Technical_Review_2026-05-12.docx` — colleague's code review (P0 bug list)
- `docs/external_review_2026-05-12_enterprise.md` — colleague's friend's enterprise strategy review (governance + serving + frontier comparison)
