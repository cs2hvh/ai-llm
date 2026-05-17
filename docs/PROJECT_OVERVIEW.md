# MyLLM — Project Index

> Quick orientation for anyone landing in `docs/`. For depth, jump to
> the appropriate doc below.

**Project**: a 1B-parameter from-scratch decoder-only foundation model on a 13-source multilingual corpus, with a distillation-augmented decay phase. Pilot (250M, val_ppl 15.34) done; Stage 2 rehearsal (1B at 10-30B tokens) is the next paid commitment.

**Lead**: harshit.hv@samatva.com (solo) · **Repo**: github.com/cs2hvh/ai-llm (`main`)

---

## What to read for what

| If you want to… | Read this |
|---|---|
| **Pick up work mid-project** (current state, immediate plan, what to NOT redo) | [SESSION_HANDOFF.md](SESSION_HANDOFF.md) |
| **Understand the design** (architecture, data flow, algorithms, sharding, eval) | [DESIGN.md](DESIGN.md) — 11 sections, 20 inline Mermaid diagrams |
| **Read the latest reviewer packet** (post-pilot, with C3 results) | [review/POST_PILOT_REVIEW_2026-05-15.md](review/POST_PILOT_REVIEW_2026-05-15.md) |
| **See the pilot's frozen artifacts** (configs, command log, R2 paths, results) | [`pilots/250m_v1/`](../pilots/250m_v1/) |
| **Understand muP scaling** (HP-transfer math, per-model width_mult table) | [mup_design.md](mup_design.md) |
| **See the Stage 3 distillation plan** (teacher choice, decay-phase mixing, license/vocab review) | [teacher_distillation_strategy.md](teacher_distillation_strategy.md) + [teacher_logit_cache_format.md](teacher_logit_cache_format.md) |
| **See the Stage 3 corpus scale plan** (Rust migration) | [stage3_rust_migration_plan.md](stage3_rust_migration_plan.md) |
| **Look at governance** (model card, data card, license register) | [governance/](governance/) |
| **Read the safety policy** | [safety_policy.md](safety_policy.md) |
| **See historical / superseded docs** (pre-pilot reviews, old plans) | [archive/](archive/) |

## Status snapshot (2026-05-17)

| Layer | Status |
|---|---|
| Pilot 250M | ✅ DONE — val_ppl 15.34 on R2 |
| FSDP gauntlet | ✅ Proven (G1-G4 + G6 cross-mesh restore) |
| Phase 1 engineering | ✅ Shipped (multi-epoch reader, --production, FSDP-safe eval, G6 tests, per-source val loss) |
| Reviewer R1+R2 | ✅ 11/13 P0/P1s closed |
| C1 per-source PPL | ✅ Banked on R2 |
| C2 throughput | ✅ Done @ chunked-CE; ⚠️ needs full-CE re-bench (bug D8) |
| C3 μP/LR sweep | ✅ Done. **peak_lr=3e-4 wins. muP transfer 250M → 1B confirmed.** |
| Stage 2 launch | 🔄 Ready (hardware/seq/budget decision pending) |
| Stage 3 base run | ⏳ Blocked on Stage 2 |
| Suite | 739 passed, 1 skipped |

For the live state with all the details, see [SESSION_HANDOFF.md](SESSION_HANDOFF.md).
