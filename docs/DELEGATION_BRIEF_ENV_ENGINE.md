# Delegation Brief — Verified-Agent Environments & Agentic Data Engine (EA lane)

**For:** the parallel team member · **From:** technical track (Claude/ST) · **Date:** 2026-07-23
**Revision: v1.1** (2026-07-23) — amended per the SESSION_SECOND audit; see §8 changelog.
**Process:** you research → we review together → your **final plan becomes a human-signed ADR**
→ build. This brief gives mission, architecture, interfaces, research questions, and
deliverables — deliberately opinionated so you have something concrete to push back on.
Nothing here is frozen.

**Authority status (v1.1 correction):** this brief authorizes **D1 research + D2 draft-planning
only**. The boundaries in §2 come from plans that are *adopted by owner direction with caps and
role-naming still pending formal signature* (G0-M). D3 prototype and D4/F1 implementation are
**HOLD** until the SESSION_SECOND §4 preconditions clear (named human EA + security/gate
reviewers, caps signed, canonical plans on main, sama-7b remote + branch protection, approved
source/image set, lane sub-budget).

**Ownership (v1.1 correction):** a **named human** holds the EA role and owns HAR-20; the ADR
and all gate evidence carry human signatures. AI sessions (this one and the EA-lane session)
draft, research, test, and provide **peer QA** — cross-checks between AI sessions are advisory
and never satisfy the independent-review requirement; a qualified non-author human (plus a
security reviewer for containment claims) is the reviewer of record.

---

## 1. Mission (why this lane matters more than it looks)

You **drive** the **EA lane** — under its **named human owner** (EA role; currently unnamed,
a G0-M blocker): the sandboxed environments where the model will eventually act, the
verifiers that decide whether it acted *correctly*, and the data engine that turns environments
into training trajectories. Three reasons this is the highest-leverage parallel work:

1. **It is the program's durable moat.** Open weights leak architecture and behavior; the
   environment fleet, verifiers, reward rubrics, and trajectory corpus never ship. Both program
   plans converge on this (FINAL_PROGRAM_PLAN §7/§9; master plan v2 §13 "Workflow Genome").
2. **It is where solo/small teams stall.** Our verified research (read
   `docs/research/7b_pivot/followup_1.md` FIRST — it is a full scoping study of exactly your
   lane) found every documented agentic-RL success was a multi-person, multi-week infra effort
   *even when reusing open components*; DeepSWE crashed the Docker daemon at 512 concurrent and
   had to rewrite on K8s mid-project. Starting now, in parallel, is how we avoid that.
3. **It is long-lead and independently researchable** (v1.2: not "cleanly decoupled" —
   *research* proceeds without blocking, but *implementation* depends on the HAR-15/HAR-16
   interfaces landing, DL source approvals, IS containment sign-off, and human review). The
   interfaces to the training track are the four schemas in §4.

Phase-1 scope = the **thin foundation** (exec plan P1-70, Linear **HAR-20**): control contract +
5 adapters + verified trajectories. Your research should also produce the **Phase-3/4 growth
plan** (RL-scale fleet, Workflow Genome) so the thin version is built on the right skeleton.

## 2. Hard boundaries (non-negotiable; source plans adopted, caps pending signature)

- **Security envelope** (exec plan P1-20/P1-70; master plan v2 §12 MCP supply-chain contract):
  per-run isolation; no production credentials —
  synthetic secrets + short-lived tokens; egress deny-by-default with allowlists; pinned
  images/commits + SBOM + malware/secret scans; resource/time quotas; fleet kill switch;
  authority broker **outside** model control.
- **No GSTN / ONDC / Tally / financial / customer systems** without separate written authority
  — documented sandbox APIs + synthetic data only. (The Indic/Indian-SaaS environments are a
  *design doc* in Phase 1, not an implementation.)
- **Public MCP server ≠ approved, and discovery ≠ acquisition (v1.1).** The "harvest" step
  stores **URLs + metadata only**. Downloading code/images/schemas, executing tools, or
  generating training data requires the full chain: source approval → quarantine → admission →
  decontamination → authorized use. Every tool that eventually enters the fleet needs
  license/API-terms review, pinned source commit, SBOM, schema signing, static + dynamic scan
  in isolation.
- **Containment is a designed boundary, not Kubernetes (v1.1).** K8s is orchestration. Before
  any D3 execution: a written **threat model** and an explicit isolation choice — dedicated
  sandbox trust domain; rootless containers; seccomp/AppArmor; gVisor/Kata/Firecracker where the
  threat model requires; no Docker socket; no host mounts; default-deny NetworkPolicy; scoped
  broker-minted synthetic credentials; kill switch. IS owns/validates this; it is D1 homework.
- **Core F1 acceptance** (do not weaken; stretch goals never displace it):
  ≥20 deterministic smoke tasks per family · **20 concurrent rollouts × 1 h** at ≥99% launch /
  ≥95% platform-completion availability (excluding injected task failures) · **zero** isolation
  breach / credential exposure / unauthorized egress / cross-run state leak · **100 verified
  end-to-end trajectories** with state-delta + outcome checks · injected timeout / tool-failure /
  stale-state / unauthorized-action / rollback cases pass · a measured report on the path to
  200-concurrent / 1,000 trajectories (stretch).
- Storage within the Phase-1 caps (50 TB core; a full SWE-rebench image mirror is ~TB-scale —
  needs its own sizing + approval line in your plan).
- Trajectories that become *training data* must flow through the corpus admission contract and
  signed token ledger like any other source — no side doors.

## 3. Proposed architecture (starting point — challenge it)

```
┌─────────────────────────── ENVIRONMENT CONTROL PLANE ───────────────────────────┐
│ Adapter contract (ONE interface, five implementations):                          │
│   init(seed)→deterministic state · capabilities() · step(action)→obs             │
│   predicted_delta vs attested_delta · audit events · timeout/quota · replay(log) │
│   reversibility class: rollback | compensating | read-only reset | prohibited    │
│ Controller (PROPOSED: K8s orchestration — D1 decides; containment boundary is a  │
│   separate D1 threat-model choice): pod pool, image pre-pull + registry + GC,    │
│   per-rollout timeouts, retry/quarantine for flaky tasks, metrics, kill switch   │
└──────────────────────────────────────────────────────────────────────────────────┘
   │ adapters (reuse-first — build almost nothing from scratch)
   ├─ repo/git ............ SWE-rebench pre-built images (21k tasks/7.5k images),
   │                        R2E-Gym (8.1k tasks); OpenHands runtime as scaffold
   ├─ terminal/fs ......... Harbor / Terminal-Bench harness (parallel cloud rollouts)
   ├─ browser/search ...... synthetic/approved targets only
   ├─ documents/sheets .... deterministic doc fixtures + state diffing
   └─ SQL/business-state .. tau2-bench Gymnasium env (200 tasks) as template
   │
┌─────────────────────────── TRAJECTORY & REWARD PLANE ───────────────────────────┐
│ Typed trajectory schema (steps, tool calls, observations, state deltas, outcome) │
│ Verifiers: final-state checks, HIDDEN tests (never visible to the agent),        │
│   schema validation; SandboxFusion for single-shot code rewards                  │
│ Anti-reward-hacking: compact filtering (mask limit-hit trajectories), block      │
│   test-file edits, decontaminate vs eval repos, pass^k for nondeterminism        │
│ Rubric-judge rejection sampling (K2 pattern) for open-ended tasks                │
└──────────────────────────────────────────────────────────────────────────────────┘
   │
┌─────────────────────────────── DATA ENGINE v0 ──────────────────────────────────┐
│ MCP tool harvest (3k+ public, supply-chain contract) → synthetic tool evolution  │
│ → task/agent/rubric generation → sim + real-exec trajectories → rubric filter    │
│ → admission pipeline → ledger-signed training shards                             │
│ Phase-1 CORE: pipeline proven end-to-end (quality bar, small volume).            │
│ ~1k filtered trajectories = STRETCH. Indic/Indian-SaaS envs: DESIGN DOC only.    │
└──────────────────────────────────────────────────────────────────────────────────┘
```

Later consumers (design for them, don't build them): SFT/mid-training data (Stage 2.5),
RLVR rollout API (verl/SkyRL/NeMo-RL-compatible), the 250k long-horizon private eval suite.

## 4. Interfaces to the training track (the contracts that keep us parallel)

| Contract | Where | Direction |
|---|---|---|
| Trajectory/rollout schema (v1.1) | **separate versioned schema you propose** (do NOT overload the training `run_manifest` — it stays a training-run contract) + a small shared **provenance envelope** common to both; human-signed ADR | you → training data + evals |
| Provenance envelope | extracted from `sama-7b/schemas/run_manifest.schema.json` commons (hashes, pins, label) | shared |
| Admission + signed ledger | HAR-15 (ST-lane builds the code; **DL role owns** approval/admission/removal — EA delivers *pre-admission trajectory packages*, never self-admits) | your packages → DL gate |
| Eval-registry fingerprints | HAR-16 — interface will be **formally versioned** before you depend on it; task/generator fingerprints exported BEFORE any trajectory becomes training data; ≥20% of task families held out **by generator family** | you → me |
| AGENTS.md rules | `sama-7b/AGENTS.md` | binding on both of us |
| Peer QA (v1.1) | EA session cross-checks eval/scorer work; ST session cross-checks env/verifier work — **advisory only**; the reviewer of record for any gate-critical claim is a qualified non-author human (+ security reviewer for containment) | mutual, advisory |

**RACI (v1.1, per exec plan):** EA = environment contracts, adapters, task generation,
verifiers, pre-admission trajectory packages. DL = source approval, quarantine/admission,
decontamination, removal lineage, signed ledgers. IS = containment, secrets, network, registry,
incident controls. ST = training integration.

## 5. Research questions your final plan must answer

**Stack (with evidence, not vibes):**
1. Trainer-side env framework: SkyRL (skyrl-gym/agent, Harbor wired) vs verl (+SandboxFusion)
   vs plain OpenEnv/verifiers contract now + trainer chosen later? Weigh: our S0 harness is
   likely Megatron Bridge; NeMo-RL adjacency may matter. Recommend one contract that survives
   any trainer choice.
2. K8s self-hosted vs rented sandbox fleets (Prime Intellect / E2B / Modal / Daytona) for
   Phase 1 vs RL-scale: cost model at 20 / 200 / 2,000 concurrent (research corpus has verified
   pricing), ops burden, data-residency (trajectories are IP — where may they live?).
3. Image logistics: mirror subset strategy for SWE-rebench (which 2k of 7.5k images?), local
   registry + GC design, storage line-item vs the 50 TB cap.
4. Determinism + replay: how each family achieves bit-reproducible init and faithful replay
   (fs snapshot? git reset? DB dump/restore? container layer reset?); what "attested state
   delta" concretely is per family (git diff, fs manifest diff, SQL dump diff, DOM diff).
5. Concurrency architecture: what actually breaks between 20 → 200 rollouts (the DeepSWE
   lesson); pod-pool vs Firecracker microVMs; where the authority broker sits.

**Verification & rewards:**
6. Per-family verifier design + the hidden-test protocol; how we measure verifier quality
   itself (target: ≥99.5% agreement with human audit, ≤0.1% false-reward — master plan §11.3).
7. Anti-hacking hygiene checklist as *code* (compact filtering, edit-blocks, decontam) and an
   adversarial verifier-test suite.
8. Nondeterminism policy: pass^k sampling, flaky-task quarantine thresholds.

**Data engine:**
9. MCP harvest pipeline: source list, supply-chain gate implementation, how 3k tools → 20k
   synthetic (K2's hierarchical domain evolution — see `docs/research/7b_pivot/agentic.md`),
   rubric-judge design, and the license audit for every reused corpus (Toucan-1.5M license
   text verification is still OPEN from our research — close it).
10. Trajectory schema: align with (or justify diverging from) Toucan / daVinci-Dev / OpenHands
    formats so public data and our engine output are unifiable.
11. Perturbed-trace generation (tool-error-recovery objective, Lane-4 IP candidate): design the
    failure-injection taxonomy. **(v1.1) Invention-candidate design notes live in the restricted
    vault only** — git/Linear/Notion/session files carry sanitized requirements + controlled
    references, never the disclosure content.
12. Indic/Indian-SaaS environment design doc: which sandbox APIs exist (Tally dev sandbox?
    GSTN sandbox? ONDC staging?), synthetic-data strategy, legal prerequisites — design only.

**Program fit:**
13. Milestone map onto F1 (what lands wk 3/5/7/8) + effort honesty (our research says 2–4 wks
    to first stable rollout loop even reusing everything — plan against that, not optimism).
14. Budget: CPU-core sizing (research anchor: ~16–32 cores per training GPU at RL time — what
    does Phase 1 actually need?), storage, any rented-sandbox spend within caps.
15. Risk register additions for your lane (escape/leak = existing R15; add your own).

## 6. Read-first list (all in this repo)

1. `docs/research/7b_pivot/followup_1.md` — **your lane's scoping study** (frameworks, costs,
   the DeepSWE/K2/Qwen evidence, build-vs-reuse map). Start here.
2. `docs/research/7b_pivot/agentic.md` — post-training recipes, tool-use datasets + licenses.
3. `docs/PHASE_1_G0_S0_EXECUTION_PLAN.md` §P1-70 + §12 — your gate + the supply-chain contract.
4. `docs/PHASE_1_PLAN.md` v2 — T8 + how your lane threads through the technical track.
5. `docs/FINAL_PROGRAM_PLAN.md` §7–§9 — where your outputs land in the whole program.
6. `docs/research/7b_pivot/w2_novel_algos.md` §4 + `w2_positioning.md` — why the training-method
   and eval IP sits in your lane's data.
7. `sama-7b/AGENTS.md` + `sama-7b/schemas/run_manifest.schema.json` — the binding contracts.

## 7. Deliverables & checkpoints

| # | Deliverable | Target | Review |
|---|---|---|---|
| D1 | **Research memo**: answers to §5 with primary sources; license audit table; cost model | ~1 week | async comments (Harshit + Claude) |
| D2 | **Final plan** (architecture chosen, milestones→F1, budget, risks) | +3–4 days | live review → **ADR-0xx signed**; Linear children created under HAR-20 |
| D3 | **Prototype**: control contract + 1 adapter (recommend repo/git via SWE-rebench subset) + 20 deterministic tasks + replay demo + 1 injected-failure case | ~wk 4–5 | demo + **human reviewer-of-record** (named at G0-M) with Claude peer QA; IS/security review of containment claims |
| D4 | **F1 acceptance**: full §2 core criteria (incl. **100** verified trajectories) + growth report. **(v1.1) The 1k-trajectory data-engine smoke is STRETCH — it informs Stage-2.5 readiness and never blocks F1** | wk 7–8 | F1 gate packet (`gates/F1/environments/`) |

**Working agreements:** HAR-20 is owned by the named human EA; you drive it day-to-day
(split into 1–5-day children after D2);
status lands in `docs/SESSION_MAIN.md` §7 like everyone's; anything you want from the training
track goes in SESSION_MAIN §5 Redirections or a Linear comment; privileged material (counsel,
partner data) never enters this repo/Linear/Notion.

*Push back freely in D1 — especially on the framework choice and the K8s-vs-rented question.
The architecture above is a strong prior from verified research, not a decision. The
SESSION_SECOND observation that our internal contract must not depend directly on OpenEnv's
stability (experimental, APIs-subject-to-change) is accepted: the likely direction is a small
framework-neutral internal contract with thin adapters toward OpenEnv/SkyRL/Harbor.*

---

## 8a. v1.2 changelog (2026-07-23, second-pass audit) — remaining contradictions closed

Mission/working-agreements "own" wording → human-EA-owns/you-drive (§1, §7); D3 review column
→ human reviewer-of-record + IS security review, Claude = peer QA; data-engine diagram 1k →
STRETCH; security citation corrected to exec-plan P1-20/P1-70 + master-plan v2 §12; "cleanly
decoupled" → independently-researchable-with-implementation-dependencies; controller K8s
labeled PROPOSED/D1-decides.

## 8. v1.1 changelog (2026-07-23) — SESSION_SECOND audit accepted in full

All nine findings of `docs/SESSION_SECOND.md` §1 accepted:
(1) human EA owns the lane; AI cross-checks are advisory peer QA, humans are reviewers of
record; (2) "signed plans" → adopted-pending-signature; authority scoped to D1/D2, D3/D4 HOLD;
(3) branch problem acknowledged — plans/research published to `main` same day (cherry-picked);
sama-7b remote remains an owner action; (4) 100 verified trajectories = F1 core; 1k engine
smoke = stretch (the brief had conflated the T6 engine target with the T8 gate); (5) RACI
tightened (EA/DL/IS/ST split); (6) discovery ≠ acquisition — metadata-first harvest;
(7) threat model + explicit isolation choice required before any D3 execution; K8s is not a
security boundary; (8) trajectory schema separated from the training run-manifest via a shared
provenance envelope; HAR-15/16 interfaces to be formally versioned before dependence;
(9) invention notes restricted to the vault.
