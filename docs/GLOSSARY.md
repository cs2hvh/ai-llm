# GLOSSARY — every code and shorthand in the SAMA-7B program, in plain language

> New to the docs? Read this first. The grammar of every task is:
> **Gates decide when we may proceed → T-tracks are the work areas → HAR-xx are the
> tickets → D-numbers record decisions → R-numbers watch risks.**

## Gates (checkpoints — doors that open only with evidence)

| Code | Name | Plain meaning | Target |
|---|---|---|---|
| **G0-M** | Mobilization ("M" = mobilization) | "May bounded setup work begin?" Small caps (600 GPU-h / 20 TB / $25k) until G0. We are here now. | day 3–5 |
| **G0** | Program authorization | "Is this a real, fundable program?" Needs the 10-interviews / 3-prototypes / 2-pilot-letters product proof, budgets, counsel, trusted measurement. | 2026-08-20 earliest |
| **S0** | Systems selection | "Which ONE training framework — Megatron Bridge or TorchTitan?" Decided by pre-registered hard gates + scorecard → ADR-013. | 2026-09-17 earliest |
| **F1** | Foundation complete | "Are data, tokenizer, evaluation, and environment foundations ready?" | 2026-09-17 earliest |
| **Phase 2** | Architecture ladder | The real experiments (150M → 350M → 1.3B → 7B rehearsal → 6T run). Opens only when **G0 + S0 + F1 all pass**. | after gates |

Rule: **dates move; gate criteria never weaken.**

## T0–T9 — technical work areas (Phase-1 technical track, `PHASE_1_PLAN.md` v2)

| Code | Work area (plain) | Main tickets |
|---|---|---|
| T0 | Fresh program repo, CI, freeze the old repo safely | HAR-7, HAR-8 |
| T1 | Fix the measurement math so every number is trustworthy | HAR-9 |
| T2 | Tiny reference models ("oracles") that prove the math of the architecture | HAR-22 |
| T3 | Small-GPU smoke ladder: 1 GPU → 8 → 2 nodes; resume; low-precision; long-context proxy | HAR-23 |
| T4 | Qualify the B200/B300 hardware + honest speed baselines | HAR-10 |
| T5 | The S0 framework bake-off itself | HAR-13, HAR-21, HAR-24, HAR-29 |
| T6 | Data & tokenizer: eval-list freeze → decontamination → small clean corpus → 3 tokenizer candidates | HAR-15, HAR-16, HAR-17, HAR-19 |
| T7 | Evaluation: fail-closed scorers, private test suites, rival-model baselines | HAR-11, HAR-25, HAR-26, HAR-27 |
| T8 | Safe agent sandboxes (delegated to the EA lane) | HAR-20 |
| T9 | The end-to-end practice pretrain (150–200M) proving the whole factory | HAR-28 |

## P0 / P1 / P2 — payload sizes inside the S0 bake-off

- **P0** = `<20M` params, CPU, exact-math oracle (the T2 models)
- **P1** = 50–200M, GPU smoke tests (parity, resume, MXFP8, context-parallel proxy)
- **P2** = full 7B shape: dense speed profile @8K/32K + hybrid feasibility @262K on one B300

(Do not confuse with the exec plan's **P1-00…P1-70**, which are that document's own
work-package numbers for the org lane: P1-10 = product proof, P1-20 = legal, P1-70 = environments.)

## D1–D13 — decisions (FINAL_PROGRAM_PLAN §14 + Notion "Decision Register")

Numbered choices with status (proposed / adopted / signed). Frequently referenced:
**D1** name+trademark · **D2** Indic-data share (evidence-ledger-gated 8–12%) ·
**D3** license (revenue-gated open weights) · **D4** token budget (6T/8T/10T) ·
**D10** architecture program (public 3:1 hybrid + our training objectives) ·
**D11** framework (Bridge default, settled at S0).

⚠️ **Collision warning**: the delegation brief's deliverables are also lettered D1–D4
(research memo → final plan → prototype → F1 acceptance). Those are **EA-D1…EA-D4**
(deliverables), not decisions.

## Other codes

| Code | Meaning |
|---|---|
| **HAR-xx** | Linear ticket ID (team prefix "HAR"). The unit of actual work. |
| **A1–A4** | The four adopted amendments from the execution plan (gate model; hard 10/3/2 product evidence; cluster-parity deferral; **A4 = the spending caps, pending owner signature**). |
| **R01–R18** | Risk register entries (Notion). E.g. R05 accounting trust, R06 B300 unreserved, R17 hiring slip. Red risks block gates. |
| **ADR** | Architecture Decision Record — a one-page "we decided X because Y" file in `docs/decisions/`. ADR-000 = mobilization; ADR-013 = the future S0 framework choice. |
| **FO / PO / ST / DL / EA / IS / FP / LC / IR / CR / LR** | Role hats (one person may wear several): Founder-owner · Product lead · ML-systems lead · Data lead · Evals+environments lead · Infra/security · Finance/people · Legal counsel · Independent technical reviewer · Customer-evidence reviewer · Language reviewers. |
| **DoR / DoD** | Definition of Ready (ticket may start: owner + acceptance criteria + estimate) / Definition of Done (tested + artifacts pinned + reproduced/reviewed by a non-author). |
| **SLV-xxx** | Salvage-register entries — the only legal path for anything reused from the legacy repo. |
| **`diagnostic`** | Label on every Phase-1 run: it proves plumbing, never architecture. Architecture claims only come from Phase-2 pre-registered runs. |
| **T2/T6… vs D2 vs P2** | Same digits, different families: **T** = work area, **D** = decision, **P** = payload size, **HAR** = ticket, **R** = risk, **A** = amendment. Read the letter first. |

## How to read any ticket (worked example)

> **HAR-9 · "T1: fix benchmark accounting" · Gate: G0 · blocks HAR-10**

= Linear ticket 9 · belongs to work area T1 (measurement trust) · must be done for the
Aug-20 program-authorization gate · and the baselines ticket (HAR-10) cannot start until
it's finished.
