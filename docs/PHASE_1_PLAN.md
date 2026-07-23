# SAMA-7B — Phase 1 TECHNICAL TRACK (v2, redesigned 2026-07-23)

> **v2 supersedes v1 of this file.** The organizational/product/legal lane (G0-M mobilization,
> hiring, counsel, design partners, budgets, RACI, gate ceremony) is owned by
> [PHASE_1_G0_S0_EXECUTION_PLAN.md](PHASE_1_G0_S0_EXECUTION_PLAN.md) (P1-00, P1-10, P1-20) and is
> **not duplicated here**. This document is the **technical execution track**: everything that
> builds and proves the training factory — repo, accounting, reference oracles, smoke ladder,
> S0 systems campaign, data/tokenizer/eval foundations, thin environments, and the end-to-end
> smoke pretrain. Owner of record for this track: ST role (interim: Harshit + Claude).
>
> **Adopted from the execution plan** (formal ADR at G0-M): the G0-M → G0 → S0 ∥ F1 gate model;
> Phase 2 blocked on G0+S0+F1; hard 10/3/2 product evidence at G0 (their lane); full-allocated-
> cluster parity deferred to the 7B-rehearsal gate; caps — **2,850 GPU-hours total (600 under
> G0-M), ~50 TB core storage, external cash per their §10**; Definition of Ready/Done; the
> measured-quantities-primary accounting doctrine; "smallest valid proof first."

---

## 0. Operating rules for this track

1. **Smoke first, always.** CPU reference (<20M, FP32) → single-GPU smoke (50–200M) → 8-GPU node
   → 2-node → CP proxy → one-node B300 full-width feasibility. Large hardware is never a
   substitute for a precise small experiment.
2. **Nothing here is architecture evidence** (their R18). Every run in Phase 1 is labeled
   `diagnostic` in its manifest. Architecture, optimizer, objective, and mix decisions happen in
   Phase 2 from pre-registered runs only.
3. **Accounting before performance.** No throughput/MFU/cost number is published before the
   accounting regression suite passes (T1). Measured step time and token counts are primary;
   MFU is secondary with a versioned FLOP model (6N is invalid as sole model at 128K/262K).
4. **Fail-closed evaluation.** A scorer or benchmark failure returns non-zero and blocks; the
   legacy "non-empty output = pass" placeholder class is banned from any evidence.
5. **Data behind controls.** Eval-registry freeze → fingerprint export → (capped) quarantine →
   admission → decontamination → tokenizer/corpus. No mass mirroring in Phase 1 (50 TB core cap);
   the Phase-2 proxy corpus is 5B unique tokens, not 10–15T.
6. **GPU-hour sub-budgets per campaign with stop rules**; failed tests are re-investigated at the
   smallest payload before any full-width rerun.

## 1. Workstreams (T0–T9)

### T0 — Program repo & controls  *(wk 1; gates G0)*
Fresh private PyTorch lineage per their §P1-32 layout (`src/sama/`, `configs/`, `schemas/`,
`containers/`, `cluster/slurm/`, `cluster/k8s-envs/`, `tests/{unit,contract,distributed,gpu,evals}`,
`evals/registry/`, `docs/decisions/`, `gates/`, `AGENTS.md`). Fresh history; pinned container +
lockfile; no imports from the legacy JAX tree except through a **salvage manifest** (source
commit/hash, destination hash, license, reviewer, reason). CI tier 1 on every PR: format/type,
unit/contract, **<20M FP32 CPU reference with toy vocab, <10 min**, token-accounting tests,
manifest schemas, scorer fixtures, secret/license scans. Self-hosted B200 tier (50–200M CUDA
smoke, 1–2-GPU parity, ckpt/resume) and nightly 8-GPU tier follow in wk 2–3.
**Legacy**: annotated tag `legacy-pilot-pre-audit-00a825f` (remote-verified) → accounting errata
applied without touching raw logs → second tag `legacy-pilot-audit-corrected` → freeze.

### T1 — Accounting correction (blocks everything measured)  *(wk 1; gates G0)*
Implement the report schema: `global_batch_sequences, local_batch_sequences,
scheduled_compute_tokens, nonpadding_model_tokens, loss_tokens, source_cursor_tokens,
DP/TP/PP/CP/world sizes, warmup/window/accum/cursor-delta`. Token counts taken at a unique
ownership boundary; cursor reconciliation mandatory. Regression tests: world sizes 1/2/4/8 ×
padding × masked targets × grad-accum × ≥1 CP topology. Historical C2/C3: recompute from raw
timings + cursor evidence or mark **withdrawn** (never estimated). Errata doc published.

### T2 — Reference oracles (P0)  *(wk 1–2; gates S0 hard-gate 1)*
`<20M` FP32 models, CPU + single GPU: **dense** and **GDN-class hybrid** oracles. Cover forward,
backward, causal masking, document packing boundaries, recurrent-state reset, gradient reference
(finite-difference where feasible), deterministic fixtures. These are the parity anchors for
every later precision/parallelism claim — and the first smoke of the whole program.

### T3 — Small-GPU smoke ladder (P1: 50–200M)  *(wk 2–5; gates S0 hard-gates 2–7)*
Order: **single-GPU BF16 smoke** (first GPU run) → 8-GPU one-node parity → 2-node parity →
checkpoint save/resume at arbitrary step with exact cursor restoration → topology reshard →
1–2K-step stability. Then **MXFP8 multi-seed campaign** (fixed token count, non-inferiority:
upper CI of val-loss regression within registered 0.5% relative margin) and a short **exact-8B
full-width MXFP8 smoke at 8K/32K** (so proxy dims don't hide vocab-head/TE failures). Then
**CP proxy correctness at 64K/128K/262K**. Failure drill: injected worker kill, incomplete-
checkpoint rejection, clean recovery, no skipped/duplicated committed data.

### T4 — Hardware qualification & trusted baselines  *(wk 2–4, after T1; gates G0)*
Per SKU (B200, B300): exact topology, driver/CUDA/NCCL/TE versions, clocks/ECC/Xid state,
NVLink/NVSwitch health, all-reduce/reduce-scatter/all-gather bandwidth, 2-node network behavior.
Baselines with corrected accounting, 3-run median + CoV: dense exact-8B surrogate @8K/32K
(BF16 + MXFP8) and long-context **proxy** @64K/128K/262K for CP correctness. **No dense-8B
full-attention 262K baseline** (unrepresentative quadratic burn). B200/B300 never mixed.
Reference ceiling: NVIDIA 26,006 tok/s/GPU (FP8 8B @8K, DGX B200).

### T5 — S0 systems campaign  *(wk 4–8; gate S0)*
**Pre-registration packet first** (frozen before results): candidates (Bridge @ pinned 26.06
digest; TorchTitan @ pinned commit, 2-engineer-week hybrid-integration timebox), payload ladder
P0/P1/P2-dense/P2-hybrid, complete surrogate tensor census + placeholder vocab, canonical
init checkpoint + parameter-name mapping + initial-logit parity, identical batches/optimizer/
reporting, per-gate thresholds (BF16 one-step loss Δ ≤0.5% rel, grad cosine ≥0.999; MXFP8
non-inferiority; resume follows registered trajectory ≥10 steps), scorecard point-mappings,
winner rule (Bridge wins within 10 absolute weighted points), GPU-hour caps per campaign.
**P2-hybrid feasibility**: fixed 7B-class reference hybrid, **one B300 NVLink node, CP=8,
262,144 tokens, 3–10 consecutive optimizer steps + checkpoint/resume continuation** — correctness,
not endurance (72h belongs to the rehearsal gate). Export → HF → **vLLM AND SGLang** load/generate
+ registered logit comparison. AdamW correctness required; **Muon recorded-if-available, never an
S0 pass/fail**. Output: profiles, scorecard, **ADR-013 harness selection**; loser leaves the
production tree.

### T6 — Data & tokenizer technical lane  *(wk 1–8; gate F1)*
Strict order: (1) **eval registry v0 freeze** + decontamination fingerprint export via
restricted/salted interface (never raw private prompts into the data path); (2) admission state
machine + signed-ledger implementation (policy approval is their P1-20; the code is ours);
(3) **capped** quarantine acquisition — golden paths only, 50 TB core cap, no mass mirror;
(4) admission fixtures: one document + one code repo + one synthetic trajectory admitted, one
prohibited artifact rejected with machine-readable reason, one takedown simulated across the
lineage graph; (5) contamination/quality/repetition/entropy/dedup processing (Gopher + byte-
entropy floors — the step-718 lesson); (6) tokenizer sample assembly from admitted material only;
(7) **3 byte-BPE candidates (128K/140K/152K)** + full scorecards incl. security battery
(malformed UTF-8, special-token/tool-tag injection, Indic combining marks) + ≥250M-token
representative sample scored per candidate — **no freeze in Phase 1**; (8) **5B-unique-token
Phase-2 proxy corpus** + held-out split under one immutable provisional tokenizer hash, packed
manifest with unique-token/effective-epoch accounting; (9) **loader proof at 2× first-rung
consumption for 1 hour** with cursor reconciliation; (10) Indic evidence ledger (per-language
unique tokens, epochs at 8/10/12%, diversity, contamination, fertility per candidate) +
long-document inventory (coherent >64K/128K artifacts only — concatenations don't count).

### T7 — Evaluation technical lane  *(wk 2–8; contract at G0, content at F1)*
Versioned task registry (source revision, license, split hash, prompt template, scorer, sandbox,
contamination status); typed schemas; **fail-closed** required suites; raw traces to controlled
storage; exact model/tokenizer/prompt/scorer/container hashes. **Scorer fixtures both ways**:
known-good passes AND deliberately-corrupted fails; code execution sandboxed (kills the legacy
HumanEval+/MBPP+ placeholder class). Private content v0 **core targets**: ≥100 long-context items
across 5 position bins to ~250K, 50 agent-state/state-delta cases, 50 injection cases (held out
by generator), **100 QA-reviewed Hindi/Hinglish function-calling items** (300 = stretch).
BFCL **v4** for gating (v3 legacy-labeled only). **Baseline anchor re-runs** under the pinned
contract: Qwen3.5-9B, Gemma 4 E4B + 12B, OLMo 3, Olmo-Hybrid-7B at exact approved revisions
(absence of a mandatory open anchor ⇒ Hold).

### T8 — Thin verified-environment foundation  *(wk 3–8; gate F1)*
One environment-control contract + five adapters (repo/git, terminal/fs, browser/search on
synthetic targets, documents/spreadsheets, SQL/business-state), each with deterministic init,
capability declaration, predicted vs actual state delta, audit events, timeouts, replay,
reversibility class. Security: per-run isolation, no production credentials, egress deny-by-
default, pinned images + SBOM + scans, authority broker outside model control, injection fixtures.
**Core acceptance**: ≥20 deterministic tasks/family; **20 concurrent rollouts × 1 h** with ≥99%
launch / ≥95% platform completion availability; zero isolation breach; **100 verified end-to-end
trajectories**; failure-injection cases pass. (200-concurrent/1,000-trajectory = stretch report,
not a blocker.) No GSTN/ONDC/Tally/financial/customer systems — synthetic only.

### T9 — END-TO-END SMOKE PRETRAIN (the "factory hello-world")  *(wk 6–8; feeds F1 + Phase-2 readiness)*
A **150–200M dense-control diagnostic pretrain on the 5B proxy corpus through the complete
production path**: admitted+decontaminated data → packed corpus with signed manifest → S0-winner
harness → WSD schedule, AdamW → transient+permanent checkpoints → mid-run kill + resume drill →
in-loop easy-suite + per-source PPL → eval-report → signed run manifest binding config/container/
data/tokenizer hashes. ~5B tokens, 1–2 seeds, ~8 GPUs, ≈150–300 GPU-hours inside the cap.
**Acceptance**: zero manual interventions after launch; resume reproduces the registered
trajectory; accounting reconciles exactly (cursor vs loss-tokens vs scheduled); eval report and
gate-packet artifacts generate automatically. Labeled `diagnostic` — it proves the FACTORY, and
doubles as the dry run of every Phase-2 rung mechanic (same tooling, same manifests).

## 2. Order of battle (dependencies)

```text
T0 repo → T1 accounting → T4 baselines → T5 S0 campaign → ADR-013
T2 oracles → T3 smoke ladder ↗                    ↘ (winner harness)
T6.1 eval-registry freeze → T6 decontam/admission → T6 tokenizer+proxy corpus → T9 smoke pretrain
T7 contract (uses T6.1) → T7 content + anchors ─────────────────────────────↗
T8 environments (independent) ────────────────────────────→ F1
B300 node reservation (their IS lane) → T3 CP proxy + T5 P2-hybrid
```

Hard blockers mirrored in Linear: HAR-9→HAR-10→HAR-13; registry→decontam→tokenizer/corpus;
corpus+ADR-013→smoke pretrain; B300 reservation→CP/262K items.

## 3. Week map (staffed-earliest; capacity-driven otherwise — dates move, criteria don't)

| Wk | Technical-track outputs |
|---|---|
| 1 | T0 repo+CI tier-1; legacy double-tag started; T1 accounting schema+tests; T6.1 eval-registry freeze; T2 oracle skeletons; B300 reservation requested |
| 2 | T1 errata published; T2 oracles green in CI; T3 single-GPU BF16 smoke; T4 hardware qualification starts; T7 registry/schemas |
| 3 | T3 8-GPU parity + ckpt/resume; T4 B200 dense baselines; T6 admission fixtures; T7 scorer fixtures; T8 controller + 2 adapters |
| 4 | T3 2-node parity + reshard; T4 B300 baselines + CP proxy; T5 pre-registration frozen; T6 quarantine golden paths; **G0 review (their lane) — our G0 items: T0/T1/T4/T7-contract done** |
| 5 | T5 P0/P1 BF16 both harnesses; MXFP8 campaign; T6 candidates 1–2; T7 anchors: first 3 models; T8 adapters 3–5 |
| 6 | T5 8B MXFP8 smoke + CP=8 262K proxy; T6 candidate 3 + scorecards; T9 prep (corpus build); T7 private content push |
| 7 | T5 **P2-hybrid feasibility on B300** + export/serve parity + failure drill; T6 5B proxy corpus + loader 2×; T9 launch |
| 8 | T5 scorecard + **ADR-013**; T9 complete + report; T6 Indic ledger + long-doc inventory; T8 acceptance runs; **S0 + F1 reviews**; Phase-2 pre-registration drafted |

## 4. Budget (inside the adopted caps)

| Campaign | GPU-hours (plan) |
|---|---:|
| T4 qualification + baselines | 200–450 |
| T3+T5 S0 (parity, MXFP8, CP, recovery, profiles, both harnesses) | 500–1,200 |
| T7 anchors + prototypes support | 200–450 |
| T6 data/tokenizer + T8 env GPU work | 50–150 |
| T9 smoke pretrain | 150–300 |
| Contingency (central, 20–25%) | ~190–560 |
| **Total vs cap** | **≈1,300–3,100 → managed under the 2,850 hard cap** (T9 second seed and anchor extras are the first cuts) |

Storage: 50 TB core cap (staging, quarantine, eval artifacts, run outputs, essential images);
any source mirror beyond it needs a source-specific approval.

## 5. Track-level risks (beyond their register)

| Risk | Control |
|---|---|
| GDN-hybrid oracle correctness subtleties (state reset at packing boundaries, chunked-scan grads) | T2 finite-difference + property tests before any GPU run; CP-aware tests in T3 |
| MXFP8 recipe drift between TE versions | Pin TE in the container digest; parity campaign re-runs on any bump |
| Proxy corpus too easy/contaminated → smoke pretrain "looks great" | T6 decontam ordering + held-out split + easy-suite sanity vs known 150M baselines |
| B300 window slip compresses T5 wk 7 | Reservation requested wk 1; CP proxy on 2×B200 as *diagnosis only* (never satisfies the gate) |
| Two-harness maintenance bleed past S0 | ADR-013 removes the loser from production tree same week |

## 6. Exit — what S0+F1 hand to Phase 2

One qualified harness (ADR-013) · trusted accounting + baselines · P0/P1 oracle+smoke suite in CI
· 5B proxy corpus + held-out split + 2× loader proof under a provisional tokenizer hash · 3
tokenizer candidates + scorecards · Indic ledger · fail-closed eval contract + private v0 +
anchor table · 5 thin environments + 100 verified trajectories · **one end-to-end diagnostic
pretrain that exercised every production mechanism** · Phase-2 pre-registration (payloads, seeds,
budgets, metrics, promotion + kill rules) — signed before any Phase-2 result is seen.
