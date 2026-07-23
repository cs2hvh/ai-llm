# Delegation Brief — Verified-Agent Environments & Agentic Data Engine (EA lane)

**For:** the parallel team member · **From:** technical track (Claude/ST) · **Date:** 2026-07-23
**Process:** you research → we review together → your **final plan becomes an ADR** → you build.
This brief gives mission, architecture, interfaces, research questions, and deliverables — it is
deliberately opinionated so you have something concrete to push back on. Nothing here is frozen.

---

## 1. Mission (why this lane matters more than it looks)

You own the **EA lane**: the sandboxed environments where the model will eventually act, the
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
3. **It is long-lead but cleanly decoupled.** Your interfaces to the training track are four
   schemas (§4). We can go weeks without blocking each other.

Phase-1 scope = the **thin foundation** (exec plan P1-70, Linear **HAR-20**): control contract +
5 adapters + verified trajectories. Your research should also produce the **Phase-3/4 growth
plan** (RL-scale fleet, Workflow Genome) so the thin version is built on the right skeleton.

## 2. Hard boundaries (non-negotiable, from the signed plans)

- **Security envelope** (exec plan §12): per-run isolation; no production credentials —
  synthetic secrets + short-lived tokens; egress deny-by-default with allowlists; pinned
  images/commits + SBOM + malware/secret scans; resource/time quotas; fleet kill switch;
  authority broker **outside** model control.
- **No GSTN / ONDC / Tally / financial / customer systems** without separate written authority
  — documented sandbox APIs + synthetic data only. (The Indic/Indian-SaaS environments are a
  *design doc* in Phase 1, not an implementation.)
- **Public MCP server ≠ approved.** Every tool enters the fleet only after license/API-terms
  review, pinned source commit, SBOM, schema signing, static + dynamic scan in isolation.
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
│ Controller (K8s): pod pool, image pre-pull + local registry + GC, per-rollout    │
│   timeouts, retry/quarantine for flaky tasks, metrics, kill switch               │
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
│ Phase-1 target: pipeline proven end-to-end with ~1k filtered trajectories        │
│ (quality bar, not volume). Indic/Indian-SaaS envs: DESIGN DOC only.              │
└──────────────────────────────────────────────────────────────────────────────────┘
```

Later consumers (design for them, don't build them): SFT/mid-training data (Stage 2.5),
RLVR rollout API (verl/SkyRL/NeMo-RL-compatible), the 250k long-horizon private eval suite.

## 4. Interfaces to the training track (the contracts that keep us parallel)

| Contract | Where | Direction |
|---|---|---|
| Trajectory schema | you propose; we co-sign as ADR | you → training data + evals |
| Run/rollout manifest | `sama-7b/schemas/run_manifest.schema.json` (extend, don't fork) | shared |
| Admission + signed ledger | HAR-15 implementation (mine) | your trajectories flow through it |
| Eval-registry fingerprints | HAR-16 (mine) — your task/generator fingerprints must be exported BEFORE any trajectory becomes training data; ≥20% of task families held out **by generator family** | you → me |
| AGENTS.md rules | `sama-7b/AGENTS.md` | binding on both of us |
| Cross-review | you review my eval/scorer work; I review your env/verifier work (satisfies the non-author-review rule with two people) | mutual |

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
    failure-injection taxonomy; this is a candidate invention record — keep design notes dated.
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
| D3 | **Prototype**: control contract + 1 adapter (recommend repo/git via SWE-rebench subset) + 20 deterministic tasks + replay demo + 1 injected-failure case | ~wk 4–5 | demo + code review (Claude) |
| D4 | **F1 acceptance**: full §2 core criteria + the 1k-trajectory engine smoke + growth report | wk 7–8 | F1 gate packet (`gates/F1/environments/`) |

**Working agreements:** you own Linear HAR-20 (split it into 1–5-day children after D2);
status lands in `docs/SESSION_MAIN.md` §7 like everyone's; anything you want from the training
track goes in SESSION_MAIN §5 Redirections or a Linear comment; privileged material (counsel,
partner data) never enters this repo/Linear/Notion.

*Push back freely in D1 — especially on the framework choice and the K8s-vs-rented question.
The architecture above is a strong prior from verified research, not a decision.*
