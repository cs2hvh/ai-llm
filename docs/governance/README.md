# Governance Artifacts

This directory keeps the compliance and release metadata that should survive
planning churn. It is now scoped to the current pre-2 direction, not the old
2026-05-12 review packet.

## Files

| File | Purpose | Status |
|---|---|---|
| `model_card_v1.md` | Draft model card for the current 1B-class base-model program. | LIVE DRAFT; update per release |
| `data_card_v1.md` | Draft training-data disclosure and source table. | LIVE DRAFT; update when the data mix changes |
| `license_register.md` | License and terms register for datasets, excluded sources, eval sets, and model-release posture. | LIVE; update on every source change |

## Current Planning Basis

- [`../PRE2_FINAL_PLAN_2026-05-14.md`](../PRE2_FINAL_PLAN_2026-05-14.md) is the current architecture, data, precision, hardware, and timeline decision document.
- [`../PRE2_ARCHITECTURE_DECISION.md`](../PRE2_ARCHITECTURE_DECISION.md) is the accepted pre-2 architecture decision record.
- [`../PRE2_STACK_MIGRATION_PLAN.md`](../PRE2_STACK_MIGRATION_PLAN.md) is the implementation backlog for the PyTorch/TorchTitan migration.
- [`../PROJECT_OVERVIEW.md`](../PROJECT_OVERVIEW.md) is the current pre-1 implementation overview.
- [`../safety_policy.md`](../safety_policy.md) is the project safety policy.
- [`../stage3_rust_migration_plan.md`](../stage3_rust_migration_plan.md) remains the data-pipeline acceleration plan, pending alignment with pre-2.

## Release Artifacts Still Needed

Before publishing any checkpoint, add or regenerate:

1. `eval_card_v1.md`: benchmark results, versions, prompts, harness commit, and contamination policy.
2. `risk_card_v1.md`: known failure modes, mitigations, residual risks, and release gates.
3. `model_supply_chain.md`: training run provenance, artifact hashes, signing keys, dependency pins, and checkpoint lineage.
4. `incident_response.md`: regression and safety-failure handling process.
5. EU AI Act technical-documentation packet compiled from the model card, data card, license register, eval card, and risk card.
6. India DPDP workflow note for PII handling, retention, and deletion.

## Update Rule

Do not add one-off review memos back into this folder. If a review produces a
lasting decision, fold it into the current plan, model card, data card, or
license register and cite the decision directly.
