# SAMA-7B Program — Company Ops Stack (2026-07-23)

> The tooling/workflow/documentation stack to run the 7B program like a real company project.
> Two tiers: **Lean** (now: solo + 1–2 hires) and **Scale** (if D9 lands on the 24–35-person
> plan). Claude integration status noted per tool — "plugin (OAuth)" means it is already
> installed in this Claude workspace and needs a one-time authorization in claude.ai →
> Settings → Connectors (or `/mcp` in an interactive terminal session).

---

## 1. Project planning / issue tracking

| Tool | Claude integration | Verdict |
|---|---|---|
| **Linear** | ✅ plugin (OAuth) — `productivity:linear` | **Recommended (Lean).** Fast, keyboard-first, built for small eng teams; cycles map cleanly onto our phase gates. Claude can create/update/query issues once authorized. |
| **Jira + Confluence** | ✅ plugin (OAuth) — `productivity:atlassian` | **Recommended (Scale).** If D9 → 24–35 people, Jira's ceremony pays off; below ~10 people it's drag. |
| GitHub Issues + Projects | ✅ plugin (OAuth) — `engineering:github`; `gh` CLI already works locally | **Minimum viable option.** Zero new tools; issues live next to code. Fine until multi-workstream coordination starts. |
| Asana | ✅ plugin (OAuth) — `productivity:asana` | Better for ops/marketing-heavy orgs; not eng-first. |
| ClickUp | ✅ plugin (OAuth) — `productivity:clickup` | Feature-rich but noisy; skip. |
| Monday | ✅ plugin (OAuth) — `productivity:monday` | Same; skip. |
| Notion (databases as tracker) | ✅ plugin (OAuth) — `productivity:notion` | OK for roadmap/docs, weak as an eng tracker. |
| TASKS.md (in-repo) | ✅ built-in — `productivity:task-management` skill | Already usable today with zero setup; good bridge until Linear is authorized. |

**Pick**: Linear (Lean) → Jira (Scale). Bridge with TASKS.md + GitHub Issues today.

## 2. Code, CI/CD, review

| Layer | Tool | Claude integration |
|---|---|---|
| Repo hosting | GitHub (`github.com/cs2hvh/ai-llm` + new private 7B repo) | `gh` CLI works now; GitHub plugin (OAuth) adds PR/issue MCP tools |
| CI | **GitHub Actions** + self-hosted GPU runners on the cluster (lint, unit tests, 150M smoke-train, kernel gradient checks) | **Claude Code GitHub Action** — `@claude` on PRs/issues runs Claude in CI (install via `/install-github-app` from an interactive `claude` terminal) |
| Code review | PR-based, protected `main`, required checks | Built-in: `/code-review` on working diff; `/review <PR#>`; **`/code-review ultra`** for multi-agent cloud review of a branch or PR; `/security-review` before releases |
| Release/versioning | Git tags per training milestone + signed training-ancestry manifest (master-plan §16 discipline) | Claude maintains manifests as part of run scripts |

## 3. Documentation architecture

**Source of truth stays in-repo markdown** (the existing discipline is a strength — keep it):

```
docs/
  PLAN_7B_AGENTIC.md            ← program plan (living)
  7B_AGENTIC_262K_MASTER_PLAN…  ← sibling plan (frozen, do not edit)
  research/7b_pivot/            ← verified research corpus
  OPS_STACK.md                  ← this file
  governance/                   ← model/data cards, license register, EU AI Act
  runbooks/                     ← NEW: launch, resume, incident, eval runbooks
  decisions/                    ← NEW: ADRs (one file per decision, D1–D12 style)
  inventions/                   ← NEW: dated invention records (IP §16.2; need-to-know)
SESSION_HANDOFF.md              ← per-phase live state (pattern already proven)
CLAUDE.md                       ← repo instructions for Claude sessions
```

Company-level (non-repo) docs: **Notion or Confluence** (both have plugins, OAuth) for
hiring, vendor contracts, board updates. **Slack canvases** (already connected — Claude can
create/update canvases today) for weekly status snapshots.
⚠️ `docs/inventions/` and the data-engine internals belong in the **private** new repo with
need-to-know access, not the current one, per the IP controls.

## 4. Communication & meetings

| Need | Tool | Claude status |
|---|---|---|
| Team chat | **Slack** | ✅ **already connected** in this workspace (search, send, canvases, scheduled messages) |
| Meeting notes | Fireflies | plugin (OAuth) — `product-management:fireflies`; adopt when meetings become real |
| Standups/status | Slack + built-in skills | ✅ `engineering:standup`, `operations:status-report` skills generate these from git/tracker activity |

## 5. ML-specific stack (no Claude connector needed — CLI/API driven, Claude operates them via Bash)

| Need | Tool | Notes |
|---|---|---|
| Experiment tracking | **Weights & Biases** (already used in pilot) | Keep. Claude reads/writes via `wandb` CLI/API. Self-hosted W&B or MLflow if data-sovereignty demands. |
| Artifact store | **Cloudflare R2** (existing bucket layout) | Keep — both plans agree it's the shared substrate. |
| Model registry | Private HF org (+ R2 mirrors) | Release staging, gated access for evals. |
| Data catalog/lineage | In-repo manifests (existing `CorpusManifest` discipline) + DuckDB over metadata | The provenance ledger IS the EU AI Act deliverable. |
| Cluster scheduling | Slurm or Kubernetes on the B200/B300 clusters (K8s required anyway for the RL sandbox fleet) | Claude drives via `kubectl`/`sbatch` in run scripts. |
| Monitoring/alerts | Prometheus+Grafana on-cluster; **Datadog** plugin (OAuth) if preferred managed | Training watchdog → Slack webhook alerts (Slack already connected). |
| On-call/incidents | **PagerDuty** plugin (OAuth) at Scale tier; Slack alerts at Lean tier | Built-in `engineering:incident-response` skill runs triage→comms→postmortem. |
| Secrets | 1Password/Vault + per-env `.env` (never in repo) | Claude never handles raw secrets. |
| Sandboxes (RL fleet) | Self-hosted K8s, or rented: Prime Intellect / E2B / Modal | Per plan §8; $2–8k per campaign if rented. |

## 6. Claude-native workflows (available today, no OAuth needed)

- **Scheduled routines** (`/schedule`): nightly eval-report digests, morning training-run
  health summaries, weekly status-report generation to Slack.
- **`/loop`**: babysit long training runs — poll checkpoints/W&B, alert on anomaly.
- **Multi-agent Workflows**: the deep-research/verification pattern used to build this plan;
  reusable for corpus audits, ablation triage, release red-teaming.
- **Custom repo skills** (`.claude/skills/`): to build in Phase 0 —
  `launch-run` (compose config → launch → register manifest), `eval-report`
  (checkpoint → suite → scorecard), `data-audit` (mix drift + decontam check),
  `release-gate` (runs the §16 262K certification checklist).
- **Installed plugin skills already useful**: engineering (standup, debug, deploy-checklist,
  architecture/ADR, tech-debt), product-management (write-spec, roadmap-update,
  sprint-planning), operations (runbook, risk-assessment, change-request, process-doc),
  **human-resources (interview-prep, draft-offer, onboarding — directly relevant to D9/D6
  hiring)**.
- **Also connected now**: Figma (design/diagrams), Supabase (if the product side needs a DB),
  browser automation.

## 7. Setup order (what to actually do)

**This week (Lean tier bootstrap):**
1. Authorize connectors in claude.ai → Settings → Connectors: **Linear** (or decide
   GitHub-Issues-only), **GitHub**, keep **Slack** (done). Optional: Notion.
2. Create the **new private 7B repo** (per both plans): scaffold `docs/` layout above,
   CLAUDE.md, branch protection, Actions CI skeleton, Claude GitHub App.
3. Port the IP ledger + gotchas into the new repo's founding docs; create `docs/decisions/`
   with D1–D12 as the first ADRs.
4. Stand up the Linear workspace: projects = plan phases (0–5), first cycle = Phase-0 tasks.
5. Create the Slack channels: `#run-alerts` (watchdog webhook), `#eval-reports`, `#papers`.
6. Set up W&B project + R2 paths for the new lineage.

**At Scale tier (if D9 → staffed program):** migrate Linear → Jira + Confluence, add
PagerDuty + Datadog, Fireflies for meetings, formal on-call rotation, SOC2-track access
controls on the invention/data-engine repos.

## 8. Decision needed

- **D13**: Tracker choice — Linear (recommended) vs GitHub-Issues-only vs Jira-now.
  Depends partly on D9 (team size). Everything else above has a clear default.
