# SAMA-7B Program — FINAL Plan v1.0 (2026-07-23)

> **The single authoritative program plan.** Consolidates and supersedes
> [PLAN_7B_AGENTIC.md](PLAN_7B_AGENTIC.md) (research-synthesis plan + §16 reconciliation) and
> incorporates [7B_AGENTIC_262K_MASTER_PLAN_2026-07-23.md](7B_AGENTIC_262K_MASTER_PLAN_2026-07-23.md) v2
> (kept frozen as authored). Where those documents give depth, this one gives **decisions**.
> Research evidence base: `docs/research/7b_pivot/` (30-agent verified corpus + citation audits).
>
> All previously open decisions are resolved below. The owner can veto any of them; absent a
> veto, this is what we execute.

---

## 0. Program statement

Build, from scratch, a **dense ~7B agentic model with certified native 262,144-token context**,
positioned on the compound wedge no incumbent occupies — **usable long-context agentic work +
Indic-language agentic excellence + provenance-clean on-prem deployment** — trained on company
B200/B300 clusters, released as revenue-gated open weights, with durable IP concentrated in a
**private data engine, verified-action runtime, training objectives, and category-defining
evaluations** rather than in architecture branding.

Working codename **SAMA-7B** (public name + trademark search due before any public disclosure).

---

## 1. Decision register — ALL RESOLVED

| # | Decision | RESOLUTION |
|---|---|---|
| G0 | Program scope & staffing | **Tier-2 target (12–18 people) via staged ramp.** Start now with solo + 2–4 hires (distributed-training eng, data eng first; then post-training/RL eng, eval/safety eng). Reach ~12 by pretrain start. v1 in ~14 months at plan velocity; degrades to 18–24 months if hiring stalls (solo-viable path per master plan v2 §18 remains the floor). **The 24–35-person Page-Memory program is NOT selected for v1.** |
| D1 | Name | SAMA-7B internal codename. Public name + trademark clearance = owner action before tech report/demo. |
| D2 | Indic share | **Ledger-gated band 8–12%** across Tier-1 (Hindi, Bengali, Tamil, Telugu, Marathi), admitted only if the per-language ledger shows ≤~2.5 effective epochs of verified-quality data at the chosen share; otherwise reduce share and narrow the market claim. **BFCL-Hi > 62.4 stays a committed product target regardless** (post-training + agentic data carry much of it). |
| D3 | License | **Source-available, revenue-gated open weights** (Liquid-style; free below revenue threshold). Correctly labeled "open-weight/source-available," never "open source." Data engine, environments, evals: trade secret. Counsel drafts terms (consolidated-revenue definition, affiliates, patent grant, AUP). |
| D4/D12 | Token budget | **6T committed / 8T planned / 10T contingency** under WSD; continuation decided by validation + downstream slopes at the 6T checkpoint. |
| D5 | Engram / Page Memory | **Deferred to v2 research lane.** Not in the v1 critical path. Revisit only if Tier-3 staffing materializes. |
| D8 | Kimi K3 weights (due 2026-07-27) | Check at S0. If KDA ships fully open with kernels under a permissive license, use it to inform the Lane-2 operator choice (GDN vs KDA-class). Does not change plan structure. |
| D10 | Architecture | **Production bet = Lane 2: public-art 3:1 GDN-class hybrid** (details §2) + **Lane 4: proprietary training objectives** as the committed IP bet. Dense control maintained throughout. No architecture-IP claims on public techniques. |
| D11 | Framework | **Megatron Bridge default** (native GatedDeltaNet operator with CP support — verified), **torchtitan as 4-week S0 challenger** with the optimizer criterion scoped per-harness (torchtitan has no merged distributed Muon; don't let that silently decide S0). One production harness after S0. |
| D13 | Tracker/ops | **Linear** (connected) + GitHub + Slack; ops stack per [OPS_STACK.md](OPS_STACK.md). |
| — | Optimizer | **AdamW is the mandatory control.** MuonClip is the leading candidate, admitted only via width-transfer sweep (spectral convention, no `consistent_rms` with transferred LR) + measured wall-clock win after communication. **QK-norm × QK-Clip run as a 2×2 factorial** with long-context metrics (OlmPool). Tune WD/τ; import nothing. |
| — | Fallback discipline | A failed candidate returns to the **strongest measured control** (dense or public hybrid). SWA+sinks is an experimental arm, not an automatic fallback. Nothing unvalidated enters the 6T run. |
| D7 | EU | Prepare Art. 53 artifacts + training-content summary regardless; launch-market scope decided at release gate. India DPDP implementation starts now (staged commencement). |

---

## 2. Architecture (v1 production bet)

**Envelope** (frozen only at the final proxy gate): 7.0–7.8B complete generation parameters
(≤8.0B serialized incl. detachable heads); 32–36 blocks; hidden 3,584–4,096; SwiGLU FFN;
**3:1 recurrent:global pattern** — 3× [GDN-class linear block → FFN] : 1× [global softmax
attention → FFN]; native 262,144 **combined prompt+generation budget** (default 246,144 + 16k).

**Global-attention layers**: GQA ratio, QK-norm, positional method (partial-RoPE vs NoPE vs
page-factorized — master plan §5.5 menu), gated attention (sigmoid output gate), trained sinks —
**all set by the pre-registered factorial ablations with 32k/64k long-context metrics from the
first rung** (OlmPool: these choices compound; none is frozen by default).

**Heads**: MTP head trained in, kept only if ≥1.3× end-to-end decode on the target server.
Tied-vs-untied embeddings measured (vocab table = 470–537M params at these widths).

**Recurrent-state precision**: BF16/FP32 accumulation until lower precision proven.

**Controls carried through every gate**: dense full-attention Transformer; local/global
Transformer; (public hybrid IS the bet); recurrent+local without extras. Matched data, tokenizer,
FLOPs, context distribution, and a light SFT before judging (MiniMax lesson: deficits hide
until post-SFT, long-range, at scale).

**Serving math that motivates the bet** (262k, per sequence): full-GQA bf16 ≈ 32–38.7 GiB KV;
3:1 hybrid ≈ ~9.7 GiB bf16 / ~4.8 GiB fp8 + O(100MB) state → ~4× concurrency on B200/B300.
fp8-KV validated at 256k on our own evals before production (documented error concentration).

---

## 3. Tokenizer

New **byte-level BPE**, ≥3 candidates in **128k–152k**; tiktoken-style pretokenization, digit
split, byte fallback, NFC-only (never touching fenced code/structured data); reserved 256–512
special-token block (roles, provenance, `<think>`, Hermes-style tool tags, FIM, pages, actions);
security battery (malformed UTF-8, special-token injection, Indic combining marks, adversarial
tool tags). Freeze only after fertility (code/JSON/shell within ~5% of Qwen3; Indic ≤ Sarvam-1
band), embedding-cost, downstream-proxy, and rights gates. **No transplant.** Old 131k
SPM-Unigram = evaluation reference only.

---

## 4. Data program

**Admission**: master plan v2 §7.0 **corpus admission contract** adopted verbatim — per-artifact
identity/lineage, rights (dataset-level label ≠ content clearance), synthetic provenance,
privacy/security scans, removal lineage, counsel-approved source classes, **signed token ledger**
bound to tokenizer/mix/checkpoints/NOTICE/EU summary. Decontaminate against the FULL eval suite
(incl. agentic + Indic + long-context) **before** the first corpus build. Gopher repetition
filters + byte-entropy floors day one.

**Eligible pool**: 10–15T tokens → 6–8T curriculum. Backbone: Dolma-3-class web + DCLM +
Nemotron-CC-v2 buckets (rights-reviewed) + Stack-Edu/Stack-v2 code (SPDX allowlist) +
FineMath/MegaMath + FineWeb-2/HPLT/Sangraha Indic (ledger-gated §1-D2) + FinePDFs long docs.
Sampling ranges per master plan §7.2, selected by proxy mixture ablations; code ramps toward
~25% end-state.

**Stages**: (1) broad base 4–16k with recurring 32k/128k samples; (2) skill mid-train 200–400B
at 16–32k (code/math/technical/structured); (2.5) **agentic mid-train 50–100B** — daVinci-Dev
PR corpus + Toucan-1.5M + Nemotron-Pretraining-SFT + own engine output (license audit per
source); (3) long-context 150B first hypothesis (20B@32k → 30B@64k → 40B@128k → 60B@262k),
30–50% short replay, ≥50% of max-length examples coherent artifacts, block-diagonal masks;
(4) WSD decay anneal on best data.

**Proprietary data graph** (moat): repo symbol/test/commit graphs, claim→span links, tool
schemas with preconditions/effects/permissions, workflow state graphs, counterfactual variants,
opt-in failure bank.

---

## 5. Training & systems

**Stack**: PyTorch locked. **Gate S0 (weeks 1–8)**: Megatron Bridge (pinned 26.06 digest) vs
torchtitan on the same model — BF16 loss/grad parity, 1–2k-step MXFP8 parity, CP=8 @262k,
arbitrary checkpoint/resume + reshard, optimizer correctness (scoped per-harness), train→HF
export→vLLM/SGLang serve, profiles on B200 **and** B300. One harness survives.

**Precision**: BF16 baseline → MXFP8 only after in-house parity; embeddings/norms/lm_head BF16;
K-dims %32.

**Topology** (hypotheses, rehearsed before approval): 4–32k on B200 pools (TP=PP=CP=1,
node-sharded optimizer, replicate across nodes); 64–128k CP=4–8; **262k = CP=8 inside one B300
NVLink node** (32,768 tokens/rank), DP across nodes; 512k = separately approved experiment.
B200/B300 never mixed in one job.

**Reliability**: global-token single-count invariant (unit-tested — the legacy bug class);
1→8→2-node→cluster loss parity; deterministic resume; atomic checkpoints (manifest-as-complete);
transient ckpt 30–60 min (keep 2–3) + permanent ~100B tokens; canary batches after restore;
recovery drills; full telemetry (loss, grad/state norms, max attn logits, page/source stats).

**Compute envelope** (planning ceiling 26,006 tok/s/GPU FP8-8B-B200; replace with rehearsal
measurements): 7B×6T ≈ 69–89k B200-hours (45–58 days on 64). **Program reserve 1.3–1.8×** →
~90–160k GPU-hours before RL campaigns. Storage: 0.5–1 PB data tiers; 24 TB uint32 token copy;
30–60 TB shards; 10–20 TB checkpoints; teacher cache 38.4 TB/100B tokens if used. Loader proven
at 2× planned consumption.

---

## 6. The experiment ladder (gates, budgets, kill criteria)

| Gate | Scale/Budget | Pass condition |
|---|---|---|
| Kernel/reference | 150–200M, ~5B tok, 3 seeds | parity, gradients, resume correct |
| Factorial | ~350M, ~20B tok, 3 seeds | pre-registered score (incl. 32k long-context metrics) — QK 2×2, GQA, positions, gates/sinks, MTP, objectives |
| Finalists | 1–1.3B, ~100B tok, ≥3 seeds | dense control vs hybrid vs hybrid+1-objective on real repos/long docs/agent histories; reproduced 262k gain; serving profile; post-SFT multi-hop check |
| **7B rehearsal** | full width, 10–30B tok | ≥72h continuous, arbitrary resume, mesh as production, short+32k/128k/262k mix, export→serve works, tensor census in envelope |
| **6T base** | committed | all above signed + complete data ledger |

Promotion/fallback table of master plan v2 §6 applies verbatim (base-quality ≤0.5% rel.,
code-copy ≤2 pts, long-context ≥85% retention/≥95% evidence, optimizer must beat AdamW on
wall-clock at matched quality, MTP ≥1.3× decode, provenance-banks ≥50% injection cut if that
arm runs).

---

## 7. Post-training & agent runtime

Stages: format SFT → capability SFT (verified code/math/terminal/browser/doc trajectories incl.
failure/recovery/rollback) → preference tuning (+ SecAlign++-style injection-resistance pairs) →
**gated distillation bake-off** (sequence SFT vs same-tokenizer on-policy vs cross-tokenizer
on-policy vs RLVR — on-policy logit KD requires **self-hosted teachers** on our clusters;
API teachers only for sequence-level traces) → RLVR (executable rewards) → **agent RL** in
private environments → long-context post-training (32k/128k/262k, adversarial evidence, stale
state) → continuous safety reruns after every stage.

**Environments: assembled, not built** — SWE-rebench pre-built images, R2E-Gym, tau2 gym,
Harbor/Terminal-Bench, SandboxFusion; OpenHands scaffold; K8s fleet day one (~16–32 CPU
cores per training GPU, or rented sandboxes); compact filtering + hidden tests vs reward hacking.
**Workflow Genome targets**: 25–40 env families, ≥10k task families, ~1M verified trajectories,
≥20% held out by generator family. **MCP supply-chain contract** (pinned commits, SBOM,
sandboxed, synthetic credentials; GSTN/ONDC/Tally via documented sandbox APIs + synthetic data
only, absent written authorization).

**Runtime IP**: Intent-Bound Action Certificate + deterministic authority broker + attested
state deltas + rollback; nested recall budgets (low/balanced/exact) as a product knob; model
heads advisory, broker is the boundary.

**Committed proprietary training objectives (Lane 4 — the IP bet)**: (a) context-compaction /
fold-your-own-window objective; (b) tool-error-recovery objective from perturbed traces;
(c) page/evidence-span auxiliary objectives; (d) counterfactual minimum-authority reward.
Each separately ablated; each a candidate invention disclosure.

**Teachers** (licenses verified): DeepSeek-V4 (MIT, explicit distillation OK), Qwen3.5/3.6
(Apache-2.0), GLM (MIT), gpt-oss-120b (Apache-2.0), Gemma 4 (Apache-2.0). Avoid: Llama-4
(naming clause), MiniMax M3 (restrictive), any NC dataset. Per-teacher approval contract
(master plan §11.2) applies.

---

## 8. Evaluation contract

Freeze harness revisions; **re-run all baselines (Qwen3.5-9B, Gemma 4 E4B/12B, OLMo
3/Hybrid-7B, strongest commercial 7–9B) inside our runtime** — never copy leaderboard numbers.

**Public**: MMLU-Pro, GPQA-D, BBEH, math-with-verification, IFEval/IFBench; BFCL v4; tau-family
(pass@1 + repeated-trial); LiveCodeBench/EvalPlus/BigCodeBench; RULER 8k→262k, HELMET, NoLiMa,
LV-Eval, LongBench-Pro, Oolong; injection/security suites. SWE-bench-V = legacy comparability
only (deprecated by OpenAI audit). **Primary coding gate = private rolling-repo suite.**

**Private (built = IP)**: BFCL-Indic (8–10 languages + Hinglish); 250k long-horizon agentic
suite (checkpointed partial credit); repo symbol/evidence retrieval; predicted/actual
state-delta calibration; "claimed success but did nothing" detection; position bins to 250k;
adaptive injections held out by generator.

**262K certification** (release may claim it only if): real 262k training sequences; 5 prompt
buckets ≥500 examples each with ≥85% retention of 32k aggregate; ≥95% exact evidence/citation
across bins incl. ~250k; ≤1 pt short-context regression; TTFT/throughput/cache/concurrency
published for 1/4/8 requests on the exact SKU; agent-state + action-safety gates pass.

**Anchors/targets (12-month, honest)**: BFCL v3 ≥60 · tau2 ≥45 · IFEval ≥85 · LCB ≥50 ·
AIME-25 ≥60 · RULER@256k ≥85 · NoLiMa retention ≥85%@128k · **BFCL-Hi >62.4 (beat
Gemma-3-27b)** · private long-horizon: ≥60% of short-context success at 250k.

---

## 9. IP, compliance, release

- **Trade secret** (never ships): data engine + generators, environment state graphs, reward
  weights, curriculum scheduler, private evals/red-team generators, precision recipes, mixes.
- **Patent candidates** (counsel-led, file before ANY disclosure; India CRI-2025 mapping):
  the Lane-4 objectives with demonstrated technical effect; certificate/attested-delta loop;
  budget-conditioned recall tied to risk. Patentability and FTO run separately.
- **Independent authorship**: random init, no transplant/merge, AI-SBOM, signed ancestry,
  invention records, license-tracked imports, public/confidential tracker separation.
- **EU**: baseline GPAI (≈2.5–6.3×10²³ FLOP ≪ 10²⁵). Art. 53 docs + copyright/TDM policy +
  training-content summary prepared regardless; Art. 50 output-marking assessed; enforcement
  active from 2026-08-02. **India**: DPDP staged implementation now; MeitY governance mapping.
- **Release artifacts**: model/data/eval/risk cards, SBOM, ancestry, NOTICE, red-team report,
  incident/takedown process, jurisdiction matrix signed by counsel.

---

## 10. Timeline (staged-ramp; replace with measured throughput after rehearsal)

| Period | Deliverables | Gate |
|---|---|---|
| **Wk 0–4** | G0 signed (scope=Tier-2 ramp, budget, first 2 hires opened); product contract started (10 design-partner interviews, 3 prototypes, 2 pilot letters); new private repo + policies; legacy repo tagged; private eval v0; corrected B200/B300 benchmarks | G0 |
| Wk 1–8 | **S0 bake-off** → one harness; tokenizer candidates + scorecards; first admitted shards + signed ledgers; env families v0 (5); data-engine v0 | S0 |
| Wk 5–20 | 150M → 350M factorial → 1–1.3B finalists; 262k proxy train/serve; prior-art matrices; Indic ledger → D2 share locked | Architecture freeze **or** strongest-control fallback |
| Wk 18–24 | Full-width 7B rehearsal (10–30B tok, 72h); corpus freeze; serving alpha | 7B authorization |
| Wk 24–36 | **6T base run** (8T continuation decision at curves) | Base checkpoint |
| Wk 32–46 | Skill + agentic mid-train; 32k→262k native continuation | 262K cert candidate |
| Wk 34–50 | SFT → prefs → distillation bake-off → RLVR → agent RL → safety | Agent utility + safety |
| Wk 48–56+ | Quantization/serving, independent red team, compliance + IP review, launch demos | Release candidate |

Slips degrade gracefully: hiring-limited → solo-viable path (18–24 months), same gates.

## 10a. Phase-1 amendments (adopted 2026-07-23 per owner direction; formal ADRs at G0-M)

Per [PHASE_1_G0_S0_EXECUTION_PLAN.md](PHASE_1_G0_S0_EXECUTION_PLAN.md) (accepted):
**A1** Gate model = G0-M (bounded mobilization) → G0 → S0 ∥ F1; **Phase 2 blocked on G0+S0+F1**.
**A2** Hard G0 product evidence = completed 10 interviews / 3 measured prototypes / 2 written
pilot commitments. **A3** Full allocated-cluster loss parity deferred from S0 to the
7B-rehearsal gate. **A4** Phase-1 caps: **2,850 GPU-hours** (600 under G0-M), **~50 TB core
storage** (staged; the earlier 150–400 TB mirroring assumption is withdrawn — mass acquisition
is not a Phase-1 activity; the Phase-2 proxy corpus is 5B unique tokens), external cash per
their §10. Muon is never an S0 pass/fail criterion. Execution split: org/product/legal lane =
their P1-00/10/20; **technical lane = [PHASE_1_PLAN.md](PHASE_1_PLAN.md) v2 (T0–T9)**, which
ends with an end-to-end 150–200M diagnostic smoke pretrain through the full production path.

## 11. Immediate next steps (this week — see Linear project "SAMA-7B Program")

1. Owner: veto-or-confirm §1 resolutions; open the 2 first hire reqs; name/trademark process.
2. Create private program repo; tag legacy repo (`git tag legacy-1b-pilot && git push --tags`).
3. Fix benchmark accounting bug (task chip pending) + port gotchas/IP ledger to founding docs.
4. Start S0 prep: pin 26.06 container, reproduce dense + GDN-hybrid controls in Bridge.
5. Tokenizer corpus assembly + first fertility scorecards.
6. Stand up corpus-admission contract + ledger schema; extend decontamination index with
   agentic/Indic/long-context evals.
7. Design-partner interview list (10) drafted; first 3 outreach.
8. 2026-07-27: check Kimi K3 weights release (D8).
