# SESSION_MAIN — SAMA-7B program · live status, plan & intentions

> **The living communication doc for the whole team.** Written and refreshed by Claude
> (technical-track assistant, operating under the ST role) at the end of every working
> session. Read this first to know: what just happened, what I'm doing next and *why*,
> what I'm blocked on, and how to redirect me.
>
> Companions: Linear "SAMA-7B Program" (issue-level truth) · Notion "SAMA-7B Program Hub"
> (company view, weekly status) · this doc (session-level narrative + intentions).
> Successor to the old `SESSION_HANDOFF.md` discipline (which is now frozen legacy).
>
> **Last refreshed: 2026-07-23 (session 1 — program kickoff day).**

---

## 0. TL;DR (2026-07-23)

1. **Program launched end-to-end today**: research (30 verified agents) → FINAL_PROGRAM_PLAN
   → Phase-1 two-lane split (org lane = `PHASE_1_G0_S0_EXECUTION_PLAN.md`, technical lane =
   `PHASE_1_PLAN.md` v2, T0–T9) → Linear restructured (HAR-5…30, G0/S0/F1 milestones;
   DoR fields — assignees/estimates/due dates — still incomplete) → Notion hub +
   decision/risk registers → **Phase-1 mobilization begun under owner direction; the G0-M
   caps signature (A4) is still pending — only zero-GPU / zero-spend work has been performed**.
2. **Legacy repo frozen** at tag `legacy-pilot-pre-audit-00a825f` (pushed).
3. **Accounting bug fixed** (world-size double-count in `benchmark_throughput.py`). Legacy C2
   "30%/46% MFU" becomes **~7.5%/11.5% as world-size-corrected estimates** — definitive values
   await recomputation from raw timings under a validated FLOP model (errata discipline).
   Fix + 13 passing regression tests + errata on branch **`audit/throughput-accounting-fix`** —
   **branch pushed; the PR itself is NOT yet opened** (owner opens + reviews; that review is
   the R05 independent-review gate): https://github.com/cs2hvh/ai-llm/pull/new/audit/throughput-accounting-fix
4. **Fresh program repo scaffolded** at **`ai-llm/sama-7b/`** (nested for convenience, but
   with its **own git history** — commit `c811f0a`, 14/14 tests passing; the outer repo
   .gitignores it so the lineages stay separate). Contains: AGENTS.md contract, 6-quantity
   accounting schema, eval registry v0 draft, run-manifest schema, **CI tier-1 workflow file
   (defined but unexercised — no remote, no runs yet)**, ADR-000, salvage register.
   **Needs a private GitHub remote (owner action)** for team visibility/CI.
5. **Nothing GPU-side has run.** All work so far is zero-GPU, inside the proposed G0-M caps.
   The caps themselves (**A4: 600 GPU-h / 20 TB / $25k mobilization; 2,850 GPU-h / 50 TB /
   $300k phase**) still need the owner's signature.
6. All Phase-1 runs will carry `label: diagnostic` — nothing before Phase 2's pre-registered
   experiments is architecture evidence. I will hold this line even if a result looks exciting.

## 1. Lanes and who does what

| Lane | Owner | Scope | Where tracked |
|---|---|---|---|
| **Org / product / legal** | Harshit (FO/PO) + roles per exec-plan RACI | G0-M/G0 evidence: caps, counsel, 10/3/2 design-partner proof, hires, budgets, B300 reservation | exec plan §5–§8, Linear HAR-5/6 |
| **Technical (this doc's lane)** | ST role; Claude executes under it | T0–T9: repo, accounting, oracles, smoke ladder, S0 campaign, data/tokenizer, evals, environments, diagnostic pretrain | `PHASE_1_PLAN.md` v2, Linear HAR-7…30 |

I draft, build, test, and automate. I do **not** sign gates, own risks, review my own work,
approve spend, or touch anything public — humans do. Anyone on the team can redirect me by:
(a) editing this doc's §5 "redirections" block, (b) commenting on the Linear issue, or
(c) changing `AGENTS.md` in the program repo (canonical contract — I re-read it every session).

## 2. State snapshot (2026-07-23, end of session 1)

| Item | State | Artifact |
|---|---|---|
| Legacy freeze (pre-audit tag) | ✅ pushed | `legacy-pilot-pre-audit-00a825f` |
| Accounting fix (T1 / HAR-9) | 🔍 **on PR, review = the gate** | branch `audit/throughput-accounting-fix`, `src/myllm/utils/accounting.py`, `tests/test_throughput_accounting.py` (13/13 local) |
| Program docs versioned | 🔍 same PR (commit 2) | all plans + `docs/research/7b_pivot/` |
| Fresh lineage (T0 / HAR-7) | 🟡 local only | `ai-llm/sama-7b/` @ `c811f0a` (own git, outer-ignored; 14/14 tests) — needs GitHub remote |
| Eval registry v0 (T6.1 / HAR-16) | 🟡 draft | `sama-7b/evals/registry/registry_v0_draft.yaml` |
| ADR-000 G0-M record | 🟡 active, caps unsigned | `sama-7b/docs/decisions/ADR-000-…` |
| Linear board | 🟡 restructured; DoR incomplete (assignees/estimates/due dates pending role naming) | HAR-5…30; HAR-7/8/9/16/20 In Progress |
| Notion hub + registers | ✅ live | hub page + Decision (14 rows) + Risk (18 rows) DBs |
| GPU / data / spend | ⬜ none consumed | caps ledger at 0 |

## 3. My plan & intentions (next 3–5 working sessions)

Stated ahead of time so the team can veto *before* I build. Order chosen so that everything
remains zero-GPU and review-friendly until the caps are signed and cluster access exists.

1. **T2 — P0 reference oracles (HAR-22; my next build).** Intent: in the fresh repo, a
   `<20M`-param FP32 *dense* decoder and a *GDN-class hybrid* reference, CPU-runnable, with:
   finite-difference gradient checks on every operator; property tests for **recurrent-state
   reset at document-packing boundaries** (the failure mode I most expect to bite at scale);
   causal-mask and packed-boundary tests; deterministic fixtures; wired into CI tier-1 under a
   10-minute budget. These oracles become the parity anchor for every later BF16/MXFP8/CP
   claim — S0 hard-gate 1. *Deliberately framework-thin PyTorch so both Bridge and TorchTitan
   can be compared against the same oracle.*
2. **T6.1 — eval-surface enumeration to freeze (HAR-16).** Intent: turn the draft registry
   into the complete versioned surface (exact dataset revisions + split hashes), build the
   fingerprint exporter (salvaged 8/13-gram xxhash64 concept, fresh implementation), with the
   private-suite fingerprints behind a salted interface. I will ask for explicit sign-off
   before declaring it FROZEN, because freezing wrongly blocks all corpus work.
3. **Admission state machine skeleton (HAR-15).** Intent: the six-state data contract as
   code with deny-by-default entrypoints and the golden-path fixtures (admit doc/repo/
   trajectory, reject prohibited, simulate takedown) — runnable in CI without any real data.
4. **Scorer-fixture harness (HAR-25).** Intent: registry/schemas + the both-ways fixture
   runner (good output passes, corrupted output FAILS) — retiring the legacy placeholder
   class permanently.
5. **After caps + cluster access land**: T3 single-GPU BF16 smoke → the ladder, and T4
   qualification benchmarks — nothing before.

**I will NOT, without explicit human sign-off**: consume GPU-hours; acquire data beyond
fixture scale; freeze the eval registry, any tokenizer, or any architecture choice; touch
production/customer systems; make anything public; exceed any cap; or present a Phase-1 run
as evidence for an architecture/optimizer decision.

## 4. Immediate implementation detail (T2, so reviewers can object early)

- Layout: `src/sama/oracle/{dense.py,hybrid_gdn.py,ops.py}`, `tests/unit/test_oracle_*.py`.
- Shapes: ~12M params (d=256, L=8, toy vocab 4k, seq ≤512) — small enough for
  finite-difference checks in seconds.
- Hybrid block: minimal gated-delta recurrence (public-art formulation) + 1-in-4 softmax
  attention; **no novel components** — the oracle must be boring and provably correct.
- Tests: grad-check vs numerical ∂L/∂θ per op; state-reset property (packed docs A|B must
  produce identical outputs to unpacked A,B separately); mask leakage test (future token
  perturbation changes nothing); determinism (bit-identical across two runs, fixed seed).
- Acceptance = S0 pre-registration references these fixtures + tolerances by hash.

## 5. Needed from humans (blockers) + redirections

| # | Action | Owner | Blocks |
|---|---|---|---|
| 1 | Review + merge the audit PR (link in §0.3) | any qualified reviewer ≠ author | HAR-9/8 close; trusted numbers |
| 2 | Sign/replace A4 caps (Notion Decision Register row A4) | FO | all GPU + spend |
| 3 | Create private GitHub repo `sama-7b`; then `cd ai-llm/sama-7b && git remote add origin <url> && git push -u origin main`; branch protection + runners | FO/IS | team visibility, CI |
| 4 | Reserve exact B300 node window | IS | T3 CP proxy, T5 262K proof (red risk R06) |
| 5 | Cluster credentials for me/CI | IS | T3/T4/T5 |
| 6 | Counsel engagement (2 briefs); hire reqs; partner outreach | FO/PO | G0 evidence |
| 7 | Jul 27: Kimi K3 weights/license check (HAR-12) | any | Lane-2 operator input (non-blocking) |
| 8 | **Name the human EA-lane owner + IS/security reviewer + non-author gate reviewer** (SESSION_SECOND §4.1) | FO | EA D3/D4; all gate-critical review |
| 9 | Approve EA minimal source/image set (licenses, digests, SBOMs, storage line) + lane sub-budget with stop rules (SESSION_SECOND §4.8–9) | FO/DL/IS | EA D3 |

**Redirections (team writes here, I obey next session):**
- *(empty — add bullets; I check this section first every session)*

**EA-lane status (from `SESSION_SECOND.md` — their comms surface, integrated here):**
D1 research memo = GO (tracker synced: HAR-20 In Progress, scope = D1/D2 only, D1 due
2026-07-30). D3 prototype / D4 implementation = **HOLD** until rows 2/3/8/9 above clear.
Second-pass audit (2026-07-23 evening) accepted in full → brief v1.2 + this doc's
overstatement fixes. Note for the EA session's next reconciliation pass: **finding §1.3
(plans absent from main) is now resolved** — canonical plans are on `origin/main`; the
`sama-7b` remote remains genuinely open (owner). Their audit of the delegation brief was accepted in full → brief v1.1
(see its §8 changelog). Agreed contract points: framework-neutral internal adapter contract
(no direct OpenEnv dependence); separate trajectory schema + shared provenance envelope
(training run-manifest not overloaded); EA delivers pre-admission packages, DL admits;
AI cross-checks are advisory — humans are reviewers of record.

## 6. Standing rules I operate under (summary — canonical text in AGENTS.md)

Smallest valid proof first · global tokens counted exactly once · all runs `diagnostic` ·
evals fail closed · registry-before-corpus ordering · dataset labels ≠ content clearance ·
pins everywhere (digest/commit/seed/hash in every manifest) · caps with per-campaign stop
rules · fresh lineage (salvage register is the only legacy path) · file-before-disclosure ·
humans own decisions · verify-before-locking (external claims get primary-source checks).

## 7. Session log (append-only)

### 2026-07-23 — Session 1: from research to mobilization
- Deep research program (2 waves + verification, 34 agents total; corpus in
  `docs/research/7b_pivot/`) → `FINAL_PROGRAM_PLAN.md` locked (owner-directed decisions).
- Cross-reviewed + adopted teammate docs: master plan v2 (architecture/IP discipline),
  `PHASE_1_G0_S0_EXECUTION_PLAN.md` (gate model G0-M/G0/S0/F1, caps, RACI, amendments A1–A4).
- **Found + fixed the throughput world-size double-count** (legacy C2 MFU inflated ×4);
  errata added; raw logs untouched; fix on review branch with 13 regression tests.
- Rebuilt Linear into a delivery board (parents/children, acceptance criteria, blockers);
  created Notion hub + decision/risk registers; posted first weekly status.
- Scaffolded fresh program lineage (`sama-7b`): AGENTS.md, accounting bedrock (6-quantity
  schema, 14 tests), eval registry draft, run-manifest schema, CI tier-1, ADR-000, salvage
  register. Phase 1 (G0-M scope) is running; GPU-side work correctly blocked on caps/access.
- Created this doc as the standing team-communication surface.
- Nested the program repo at `ai-llm/sama-7b/` (own git history, outer-ignored).
- **Delegated the EA lane** (verified-agent environments + data engine, T8/HAR-20) to the
  parallel team member — research brief at `docs/DELEGATION_BRIEF_ENV_ENGINE.md`; process
  D1 research memo → D2 plan review → ADR → build. Cross-review pact established (EA reviews
  eval/scorer work; ST/Claude reviews env/verifier work).

### 2026-07-23 — Session 1 (evening): second-pass audit → brief v1.2 + accuracy fixes
- EA session's second-pass review: 9 resolved items confirmed; 7 brief contradictions + 5
  SESSION_MAIN overstatements flagged. **All fixed**: brief v1.2 (§8a changelog — own→drive
  wording, human reviewer-of-record on D3, 1k→stretch in diagram, correct security citations,
  decoupling claim softened, K8s labeled proposed); this doc corrected (G0-M "begun under
  owner direction, caps pending" not "started"; "branch pushed, PR not yet opened"; MFU as
  corrected *estimates*; CI "defined but unexercised"; Linear "restructured, DoR incomplete").
- Linear HAR-20 synced with reality: In Progress, D1-scope note, due 2026-07-30, description
  citations fixed.

### 2026-07-23 — Session 1 (later): SESSION_SECOND audit integrated
- EA-lane session audited the delegation brief (`docs/SESSION_SECOND.md`) — **all 9 findings
  accepted**; brief amended to v1.1 (ownership/authority wording, discovery≠acquisition,
  threat-model-before-D3, schema separation, 100-vs-1k scope fix, RACI, vault rule).
- Fixed their branch finding: plans + research corpus **cherry-picked onto `main`** so the
  read-first set is on the authoritative branch (accounting fix stays on the review PR).
- SESSION_SECOND.md committed verbatim as the EA lane's comms surface; new owner blockers
  (name human EA/reviewers; approve source set + sub-budget) added to §5.

*(Next session appends here; §§0–5 get refreshed in place.)*
