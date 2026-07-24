# SESSION_SECOND — parallel EA-lane communication

> **Purpose:** living communication file for the second technical session working on the
> SAMA-7B verified-agent environments and agentic data-engine lane. The primary session should
> read this file before coordinating HAR-20 or changing shared contracts.
>
> **Session role:** research, draft, test, and propose. This session is not an accountable owner,
> signatory, reviewer-of-record, or gate decider. Human ownership rules in
> `sama-7b/AGENTS.md` remain binding.
>
> **Started:** 2026-07-23 · **Last refreshed:** 2026-07-23 (D1 research complete)

---

## 0. Current conclusion

**D1 is complete:** the evidence-backed memo is
`docs/research/7b_pivot/ea_lane_d1_research_memo.md`. The delegation brief is a strong research
brief, but it is not yet a safe implementation authorization.

- **GO now:** human/primary-session review of D1 and a D2 draft plan/ADR decision packet.
- **HOLD:** D3 prototype and D4/F1 implementation until the human authority, repository,
  security, source-approval, and shared-contract blockers below are resolved.
- No GPU work, bulk data/tool acquisition, sandbox execution, external spend, or production/live
  system access is authorized by this session.

## 1. Verified findings the primary session should know

1. **Human ownership and review are mandatory.** The brief's v1.2 substance now agrees with
   `sama-7b/AGENTS.md`: AI sessions may provide peer QA, but a named human EA owns the lane, a
   human signs the ADR, and a qualified non-author human/security reviewer validates gate-critical
   evidence. The actual people still need to be named.
2. **Authorization is D1/D2 only.** The brief now records the correction: owner decisions/caps
   remain pending, research and drafting may proceed, and T8 implementation requires explicit
   authorization plus named capacity.
3. **The canonical-plan branch blocker is cleared; the nested-repo blocker remains.**
   `origin/main` now contains the authoritative plan/research set. The nested `sama-7b`
   repository remains local-only with no remote. Before parallel code changes, create the private
   remote and use protected branches plus separate feature branches/worktrees.
4. **Core versus stretch is now corrected in prose.** F1 core is five adapters, at least 20
   deterministic tasks per family, the 20-concurrent one-hour reliability run, zero security
   breach, injected failure cases, and **100 verified trajectories**. The 1,000-trajectory target
   remains stretch/growth evidence and cannot block F1.
5. **RACI boundaries are now recorded.** EA covers environment contracts, adapters, task
   generation, verifiers, and pre-admission trajectory packages. DL owns source approval,
   quarantine/admission, decontamination, removal lineage, and signed ledgers. IS owns
   containment, secrets, network controls, registries, and incident controls. ST owns training
   integration.
6. **Discovery is not acquisition.** The correction is accepted, but the delegation diagram still
   needs visual/order reconciliation. Initial MCP discovery stores URLs and metadata only. Code,
   images, schemas, tool execution, or generated training data must follow source approval →
   quarantine → admission → decontamination → authorized use.
7. **Kubernetes is orchestration, not a security boundary.** The brief now requires, before D3,
   a threat model and an explicit isolation choice: dedicated sandbox trust domain, rootless
   containers, seccomp/AppArmor plus gVisor/Kata/Firecracker as appropriate, no Docker socket,
   no host mounts, default-deny network policy, scoped broker-minted synthetic credentials, and
   a kill switch.
8. **The four claimed interfaces are not implemented.** HAR-15 admission/ledger is absent,
   HAR-16 is a draft registry without a fingerprint exporter, and the current run manifest is a
   training-run schema rather than a rollout contract. Prefer a shared provenance envelope plus a
   separate versioned rollout/trajectory schema instead of overloading the training manifest.
9. **Restricted IP stays out of ordinary collaboration systems.** Perturbed-trace invention
   notes require the restricted invention vault. Git, Linear, Notion, and these session files may
   contain only sanitized requirements, decisions, and controlled references.

## 2. External research checks already completed

- DeepSWE's 512-container Docker failure, Kubernetes migration, and 1,000+ CPU-core fleet are
  supported by the official report.
- Kimi K2 documents 3,000+ real MCP tools and more than 20,000 synthetic tools; that validates the
  pattern, not the appropriateness of copying its scale into Phase 1.
- SkyRL v0.3.0, SkyRL-Gym/Agent, and the Harbor integration are real candidates.
- OpenEnv is promising and BSD-3-Clause, but its own repository labels it experimental with APIs
  subject to change. Our internal contract therefore must not depend directly on OpenEnv stability.
- Harbor supports Terminal-Bench and cloud-provider rollouts; tau2/tau3 exposes a Gymnasium
  interface. Exact revisions must be pinned because current grading revisions are not always
  comparable with earlier results.
- Toucan-1.5M is marked Apache-2.0 and its pipeline code is MIT. That does not automatically clear
  every underlying MCP specification, tool response, API term, teacher output, or dependency.
- Vendor pricing and the 16–32 CPU-cores-per-training-GPU heuristic are planning inputs, not
  validated capacity laws. D1 must recompute them at 20/200/2,000 concurrency with storage,
  egress, retries, observability, and data-residency costs.

## 3. Parallel working boundary

### This second session may work on

- D1 primary-source research and evidence table.
- Framework-neutral adapter/control-plane contract proposal.
- Threat model and containment decision matrix.
- Per-family determinism, state-delta, verifier, replay, and rollback design.
- Trajectory/provenance/rollout schema proposal.
- Rights and supply-chain audit table for candidate frameworks, datasets, images, and tools.
- Self-hosted versus managed-sandbox cost model.
- F1 milestone, risk, capacity, and stop-rule proposal.
- Sanitized D2 ADR draft for human review.

### This second session will not do yet

- Change shared schemas or `AGENTS.md`.
- Download or execute public MCP servers, SWE images, or external datasets.
- Stand up Kubernetes, managed sandboxes, credentials, registries, or cloud resources.
- Build the D3 adapter prototype.
- Modify `SESSION_MAIN.md`; the primary session remains its integrator.
- Declare an ADR signed, a review independent, or a gate passed.

### Shared-file coordination

- This file is the second session's communication surface.
- The primary session may read it and incorporate verified summaries into `SESSION_MAIN.md`.
- Proposed shared-contract changes should be described here first, then reviewed before editing
  `sama-7b/schemas/` or common interfaces.
- Once the private remote exists, this lane should use a dedicated feature branch/worktree and PR.

## 4. Inputs needed before D3

1. Name the human EA, IS/security reviewer, and non-author gate reviewer.
2. Reconcile/sign the G0-M authority and relevant caps.
3. Canonical plans are now on `origin/main` (**cleared 2026-07-23**).
4. Create the private `sama-7b` remote, protected `main`, CI, and isolated branch/worktree.
5. Reconcile the brief's stale v1.1 header and its data-engine diagram/order wording; the
   substantive ownership, scope, security, RACI, and reference corrections are already recorded
   in its v1.2 changelog.
6. Approve the rollout/trajectory contract and schema ownership boundaries.
7. Land or formally version the HAR-15 admission interface and HAR-16 fingerprint interface.
8. Approve a minimal source/image set, its licenses/API terms, container digests, SBOMs, scans,
   storage allowance, and sandbox threat model.
9. Allocate a lane sub-budget with burn accounting, stop rules, and a capacity-based schedule.

## 5. D1 handoff and next work

The D1 memo is complete at
`docs/research/7b_pivot/ea_lane_d1_research_memo.md`. It answers all 15 questions while
separating:

- Phase-1 core from later Workflow Genome growth;
- measured facts from estimates;
- orchestration from isolation;
- environment execution from training integration;
- dataset-level labels from artifact-level commercial clearance;
- deterministic initialization from faithful replay;
- rule/state verifiers from rubric-judge quality filters;
- advisory AI cross-checks from accountable human review.

Its recommended D2 direction is a SAMA-owned, framework-neutral environment protocol and
trajectory evidence bundle; Harbor is the leading first repo/terminal adapter, while
OpenEnv/NeMo Gym and verl/SkyRL/NeMo-RL remain compatibility/trainer adapters. A managed
public/synthetic F1 fleet or company-controlled BYOC for proprietary material is preferred before
a dedicated multi-cell CPU trust domain at sustained scale. No architecture selection should be
frozen until human and IS review.

The next safe deliverable is the **D2 draft plan/ADR packet**. It must resolve the protocol/schema
ownership, isolation/provider choice, verifier statistical target, minimal admitted source/image
set, human staffing, lane caps, and D3 entry criteria. It must not start D3.

## 6. Session log

### 2026-07-23 — delegation audit

- Read and audited `DELEGATION_BRIEF_ENV_ENGINE.md` against the final plan, Phase-1 technical
  track, G0/S0/F1 execution plan, master plan, research corpus, `AGENTS.md`, repository state,
  run-manifest schema, and Linear HAR-20.
- Independently cross-checked research evidence, canonical-plan/security consistency, and
  execution readiness.
- At that audit point HAR-20 was Backlog; `SESSION_MAIN.md` now records it **In Progress** with
  D1/D2-only scope and a 2026-07-30 D1 due date.
- Recorded the corrections and parallel boundary in this file. No implementation or tracker
  mutation was performed.

### 2026-07-23 — D1 research memo completed

- Completed primary-source review of OpenEnv, Harbor, SkyRL, verl, NeMo Gym/NeMo-RL,
  OpenHands, managed sandbox vendors, Kubernetes/gVisor/Firecracker controls, MCP Registry,
  candidate corpora and Indic sandbox/onboarding surfaces.
- Proposed SEP: a SAMA-owned trainer-neutral contract with independently attested state deltas,
  authority/reversibility, two replay modes, fail-closed verification, and a remotely anchored
  tamper-evident trajectory bundle.
- Recomputed 20/200/2,000 capacity and usage-cost estimates; proposed review-only $5k external,
  4 TB physical-storage high-water and 30-planned/50-max GPU-hour lane lines with stop rules.
- Demonstrated that 100 F1 trajectories cannot certify a 0.1% false-reward target: with zero
  observed false accepts, about 2,995 known-negative cases are required for a one-sided 95%
  upper bound below 0.1%.
- Corrected the data-engine order to metadata discovery → DL approval → quarantine/scans →
  admission/decontamination → simulation/execution → EA pre-admission package → DL corpus
  admission. No downloads, execution, infrastructure, spend, schema changes, tracker mutations
  or live-system access occurred.
- Advisory peer QA then tightened controller HA and externally anchored audit evidence, separated
  logical/physical/core storage ledgers, corrected Prime/image/unit assumptions, removed a replay
  quarantine contradiction, defined verifier denominators/confidence bounds, separated runtime
  execution approval from corpus admission, and added revocable raw-data indirection. This remains
  AI peer QA, not the required human/security review.

