# Project SAMA-7B — End-to-End Plan (v1, 2026-07-23)

> **The pivot plan.** From-scratch ~7–8B transformer-based agentic model with ≥250k usable
> context, original IP, trained on company B200/B300 clusters. Supersedes the 1B MyLLM
> Stage 2/3 plan (that state is preserved in [SESSION_HANDOFF.md](SESSION_HANDOFF.md)).
>
> **Provenance**: synthesized from a 30-agent deep-research program run 2026-07-23
> (12 research dimensions + 4 gap-closing follow-ups + repo salvage audit + adversarial
> verification of every load-bearing claim against primary sources). Claims below carry
> their source URLs. Anything marked ⚠️UNVERIFIED needs a check before money moves.
>
> "SAMA-7B" is a working codename only — naming/trademark is Decision D1.

---

## 0. TL;DR — the ten locked recommendations

1. **Goal**: a 7–8B agentic model that wins a *compound wedge nobody occupies*:
   **usable 250k-token agentic work + Indic-language agentic excellence + provenance-clean
   on-prem deployment**. Not "another open 7B."
2. **Architecture**: **3:1 hybrid linear-attention** (our own named gated-delta block variant
   : gated full attention with trained sinks), native-262k via 4k→32k→128k→512k curriculum,
   **MTP head** for self-speculative decoding. This is the only route that simultaneously buys
   the 250k requirement, serving economics (~4× KV reduction), and credible originality.
   Fallback (pre-registered): MiMo/gpt-oss-style SWA 5:1 + trained sinks.
3. **Stack**: **PyTorch + torchtitan** for pretraining (all at-scale Blackwell evidence is
   PyTorch; the hybrid architecture's kernels — flash-linear-attention — are PyTorch/Triton-only),
   **verl/SkyRL** for RL, **vLLM/SGLang** for serving. The JAX/Keras3 stack is retired;
   ~60% of the repo's *value* (data pipeline, governance, gotchas, tokenizer pipeline) carries over.
4. **Optimizer**: **MuonClip-style** — Muon on hidden matrices with optax/spectral scaling
   convention (the `muon_disable_mup_scale=true` finding was correct; see §6.3), AdamW on
   embeddings/norms, WD 0.1, QK-Clip τ=100 always on, QK-norm, z-loss, WSD decay-to-zero.
5. **Data**: ~**6T tokens floor** (extend toward 10T via WSD stable-phase extension if cluster
   time allows). Dolma-3-Mix backbone + Nemotron-CC-v2 + concentrated **Indic Tier-1 slice
   (~12–15%)** + ~25% code end-state + a dedicated **agentic mid-training stage** (daVinci-Dev,
   Toucan-1.5M, own synthetic engine) — the highest-leverage addition for the agentic goal.
6. **Tokenizer**: retrain — new **~131–152k byte-level BPE** (tiktoken-style pretokenizer,
   digit split, byte fallback, 256–512 reserved special tokens for chat/think/tool/FIM),
   trained on a three-way mix: code/JSON/shell + English web + Indic Tier-1. The old 131k
   SPM-Unigram file is code-naive and dies; its Indic fertility is reproduced in the new vocab.
7. **Post-training**: SFT → DPO → **on-policy distillation (replaces most RL at ~1/10 cost)** →
   RLVR polish in assembled environments (SWE-rebench images, R2E-Gym, tau2-gym, Harbor) →
   safety (Tulu-3 mix + SecAlign++ injection resistance + runtime guard). Honest calendar:
   **3–5 months** for the full agentic post-training stage.
8. **IP strategy (the "not a copy" answer)** — three layers:
   (a) **proprietary agentic data engine** (trade secret, never ships — the only moat that
   survives open weights); (b) **named architecture identity** with tech-report ablations and a
   distinct `model_type`; (c) **category-defining evals we build**: BFCL-Indic + a 250k
   long-horizon agentic suite. Plus 2–3 novel *training objectives* (context-compaction,
   tool-error-recovery) — the best publishable-IP surface for a small lab.
9. **License/monetization**: revenue-gated open weights (Liquid-AI-style: free < $10M revenue),
   private data engine, paid in-VPC deployment; governance artifacts as the compliance
   deliverable (EU AI Act enforcement begins 2026-08-02; we're baseline-GPAI only).
10. **Envelope**: main run ≈ **69–89k B200-hours** (6T tokens, 35–45% MFU) ≈ 45–58 days on
    64 B200s; wind tunnel + migration ≈ 2–3 months before that; post-training overlaps.
    **~9–12 months to v1 release.** The window matters: the Indic-agentic and usable-250k
    quadrants are open today but closable by others within 12–18 months.

---

## 1. Positioning and identity

### 1.1 The wedge (evidence-backed)

| Gap in the field (verified mid-2026) | Evidence |
|---|---|
| Long-horizon agentic work unsolved at every scale | OSWorld 2.0: best frontier 20.6%, best open models 4.6% (arXiv:2606.29537) |
| Nominal context ≠ usable context | With 100% retrieval recall, Qwen3-Coder-30B resolves only **7% of bug-fix tasks at 64k** (arXiv:2602.16069); NoLiMa: 11/13 models drop below 50% of short-context baseline by 32k (arXiv:2502.05167) |
| Indic agentic tool use weak everywhere | Best BFCL-Hi score anywhere: Gemma-3-27b at **62.42**; Sarvam-M 48.60; a targeted 4B beats Llama-3.1-405B (arXiv:2508.19831) |
| Indian sovereign models not agentic/long-context-first | Sarvam 30B = 65k ctx, 105B = 128k, no published agentic scores (sarvam.ai, Feb 2026) |
| True 7–9B dense class thin on context | Qwen3-8B 32k native; OLMo 3 65k; Gemma 4 E4B 128k. The 262k-native small models are all hybrid/MoE |

**Positioning statement**: the provenance-clean, on-prem-deployable 7B-class agentic model that
*demonstrably* works at 250k (measured task success, not config numbers) and is the best small
model for Indic-language tool use. MCP-native from day one (28% of Fortune 500 implementing MCP).
Skip computer-use/RPA as a capability — integrate into those platforms via MCP instead.

### 1.2 Flagship demos to build toward
1. **Whole-repo refactor at 250k in one session** (vs the 7%-at-64k industry bar).
2. **500-page Hindi+English contract/tender workflow** with tool calls.
3. **First-party MCP servers for Indian enterprise systems** (Tally, GSTN-adjacent, ONDC) —
   an integration surface no US/Chinese lab will build.

### 1.3 What we publish vs what stays secret
- **Publish** (credibility): tech report with fair-comparison ablations, the named architecture
  block, muP-for-hybrid transfer recipe, BFCL-Indic and the 250k agentic eval suites.
- **Trade secret** (moat): the agentic data engine (tool synthesis pipeline, trajectory corpus,
  reward rubrics), Indic curation pipeline, RL environment fleet configs.
- License: LFM-style revenue-gated open weights (D3). Trademark the name. Patents optional/defensive.

---

## 2. Model specification (to be finalized by wind tunnel — §7)

| Component | Spec | Grounding |
|---|---|---|
| Class | Dense ~7–8B (no MoE for v1; upcycle later if serving economics demand) | From-scratch MoE = routing/EP/stability risk for a small team; dense-then-upcycle proven (Qwen, NVIDIA) |
| Layout | **32–36 layers, 3:1 pattern**: 3× \[linear block → SwiGLU FFN\] : 1× \[gated full attention → FFN\] | Qwen3.5 (all sizes incl. 9B), Kimi Linear (RULER@1M 94.8, 75% KV cut, arXiv:2510.26692), Olmo-Hybrid-7B (~2× data efficiency claim), Falcon-H1R-7B (256k at exactly 7B) |
| Linear block | **Our named variant** of the gated delta rule (KDA/GDN family) — the gating/state design is the visible-IP slot. Base: flash-linear-attention chunked kernels | KDA = channel-wise-gated GDN; design space is genuinely open (Qwen GDN ≠ Kimi KDA ≠ Falcon parallel-hybrid, all shipped within 9 months) |
| Full-attention layers | GQA (ratio TBD by ablation), head_dim 128–256, **gated attention (sigmoid output gate) + trained attention sinks**, partial-RoPE (or NoPE) with **layer-selective YaRN** on these layers only. ⚠️ **QK-norm and GQA aggressiveness are wind-tunnel ablations with long-context gates, NOT frozen defaults** — Ai2's OlmPool study (verified, §16) shows QK-norm is the single largest long-context degrader (+6 HELMET@32k when removed) and stacked QK-norm+GQA+SWA choices compound up to −47% | Qwen gated-attention NeurIPS-oral (arXiv:2505.06708); gpt-oss/MiMo trained sinks; OlmPool (allenai.org/blog/olmpool, Apr 2026) |
| Context | Pretrain 8k → ABF 32k → extension 128k → **512k trained (serve 262k+)** — overshoot improves 128k quality (Nemotron-H, arXiv:2504.03624) | Long phase costs only 40–100B tokens (ProLong, arXiv:2410.02660; Granite 4.1) |
| MTP | 1-layer (or Gemma-4-style detachable drafter) trained-in; target ≥70% 2nd-token acceptance at 1B proxy, 85–90% at scale | DeepSeek-V3 (~1.8× decode TPS), MiMo-V2-Flash 27T-token validation |
| Norm | RMSNorm pre+post (sandwich), norm_eps 1e-5 | Gemma 4 / OLMo 3 consensus |
| FFN | SwiGLU, ~3.5–4× hidden | consensus |
| Embeddings | Untied, ~131–152k vocab (tied 262k would be ~15% of params — reject) | followup-2 math |
| Stability | z-loss 1e-4, QK-Clip τ=100, grad-clip 1.0. QK-norm only if it survives the OlmPool long-context ablation — working hypothesis: **QK-Clip (fires only on logit explosion) replaces always-on QK-norm**, keeping Muon's stability guarantee without QK-norm's long-context tax | K2 (QK-Clip benign over 15.5T tokens) + OlmPool (§16) |
| Optional (gated) | **Engram-style conditional memory bank** (DRAM-resident Indic/sovereign knowledge table, ~25% of capacity) — the boldest architecture-IP bet; wind-tunnel gate decides | DeepSeek Engram (arXiv:2601.07372): +3–5 pts at 27B, NIAH 84→97, <3% throughput penalty, open code; NOT yet externally replicated |

**KV-cache math at 262k (the business case)** — formula `2 × layers × kv_heads × head_dim × seq × bytes`:
full-GQA bf16 ≈ **38.7 GB/sequence** (~4 concurrent sessions per B200) vs 3:1 hybrid ≈ **9.7 GB**
(~18 sessions; ~35 with fp8-KV — validate fp8-KV at 256k: vLLM documented error accumulation there).

**Explicitly rejected for v1** (with reasons, from the noise-floor replication arXiv:2605.20798 —
only 2 of 20 post-2021 tweaks survived Bonferroni at 1.2B): differential attention, nGPT,
FoX, value residual as identity, depth-recurrence, byte-level/H-Net, MLA, from-scratch MoE,
NSA/DSA trainable sparse (defer to v2), SuperBPE (wind-tunnel lottery ticket only).

**The MiniMax M2 counter-evidence** (they reverted to full attention after hybrid deficits in
multi-hop reasoning surfaced only at scale) is the #1 architecture risk. It is retired
empirically, not by argument — see wind-tunnel gates §7. Kimi K3 (Jul 2026, 2.8T, KDA) and
Qwen3.5's full lineup are two large labs betting the other way *after* MiniMax's warning.

---

## 3. Stack decision

**PyTorch + torchtitan.** The deciding facts:
1. Every third-party at-scale Blackwell pretraining datapoint is PyTorch: Lambda 55–60% MFU on
   Llama-8B/8×B200; Crusoe 1,856-GPU MXFP8 run at 1.22–1.28× over BF16 with loss parity
   (pytorch.org blog 2025-09-03); MLPerf v5.1/v6.0.
2. The hybrid architecture forces it: flash-linear-attention (GDN/KDA kernels), SGLang's
   CuteDSL Blackwell kernels, torchtitan ring-attention CP (verified to 1M seq on Llama-8B)
   are all PyTorch. JAX has no production delta-rule kernels — staying JAX would mean writing
   them ourselves.
3. Agentic RL (verl/SkyRL) and serving (vLLM/SGLang) are PyTorch-only on GPUs. One framework
   end-to-end removes the JAX→HF conversion-fidelity risk class (the Gemma PR #29402 bugs).

**Costs accepted**: training-core rewrite (~6–12 person-weeks), distributed-Muon port
(torchtitan has NO merged distributed Muon as of 2026-07-23 — port Moonshot's ZeRO-1-style
Megatron implementation or NorMuon's FSDP2 implementation, arXiv:2510.05491; contained 2–4 week
task), fresh muP validation (required anyway — new architecture needs a new wind tunnel, and
**no public muP recipe exists for GDN/KDA layers**: deriving one is publishable IP).

**Precision**: BF16 correctness baseline → **MXFP8** (TorchAO/TE 2.16+) after an in-house
1–2k-step loss-parity A/B. Keep embeddings/norms/lm_head BF16; GEMM K dims % 32 == 0.
Expect ~1.15–1.25× at 7B (headline 1.28× was at 70B). **No NVFP4** for the core run
(NVIDIA-internal evidence only, 4 mandatory exotic techniques).

**Ops**: HSDP (shard in node, replicate across) — no TP at 7B; CP=4–8 only for ≥64k phases;
DCP async checkpoints every 30–60 min (~70–85 GB state w/ Muon hybrid) + permanent every
50–100B tokens (~6 TB retained); elastic auto-restart (torchft only if >256 GPUs/spot);
global batch 4M tokens ramping 8–16M (Muon's large-batch tolerance, arXiv:2505.02222);
streaming dataloader needs only ~6 MB/s aggregate — trivial; intra-document masking mandatory.

---

## 4. Tokenizer (Phase 0, irreversible — do this right)

- **New byte-level BPE, vocab ~131–152k** (final size set by fertility benchmarks).
  tiktoken-style regex pretokenization, digit splitting, byte fallback, NFC normalization.
- **Reserved special-token block (256–512)**: chat markup, `<think>`/`</think>`,
  Hermes-style tool tags (day-1 vLLM `--tool-call-parser hermes` + `--reasoning-parser`),
  FIM triplet, MCP markers, spare slots.
- **Training mix (three-way, deliberate)**: heavy code+JSON+shell+markup slice; English/web
  slice; **explicit Indic Tier-1 slice** (Sangraha-verified + HPLT) to reproduce the
  1.4–2.1 tokens/word Indic fertility (the portable part of the old "sovereign advantage").
- **Gates before freeze**: bytes/token on Python/JS/shell/JSON within ~5% of Qwen3;
  Indic fertility ≤ Sarvam-1 band; compression sanity on agentic trajectories.
- Old 131k SPM-Unigram: retired (fit on a code-free mix = permanent per-token tax on the
  target workload). The training/validation pipeline code is reused.

---

## 5. Data plan (~6T floor; stages, sources, licenses)

**Stage 1 — bulk pretrain, ~5–5.5T @ 8k seq.** Backbone: Dolma 3 Mix proportions
(76% filtered web / 14% science PDFs / 7% code / 3% math — all ODC-BY; note the 7B mix's
olmOCR `[REMOVED]` redactions; Ai2 steers to the 32B mix). Augment web with Nemotron-CC-v2
high-quality buckets (⚠️ NVIDIA Data Agreement needs legal read for redistribution posture).
**Indic Tier-1 concentrated slice ~12–15%** (D2): Hindi, Bengali, Tamil, Telugu, Marathi from
Sangraha-verified (CC-BY-4.0, 64B verified tokens) + HPLT 3.0 (CC0, ~70B Indic) + FineWeb-2
(ODC-BY) + MADLAD (CC-BY), ~1.5–2.5 epochs. Cross-family transfer is minimal
(arXiv:2410.12883) — a thin 22-language spread is wasted; concentrate or don't bother.
Code ramps toward ~25% end-state (optimal-share evidence, arXiv:2408.10914).

**Stage 2 — mid-train, 100–300B.** Dolmino-style: MegaMath/Nemotron-CC-Math, Stack-Edu,
QA, instruction-adjacent, reasoning traces; push code+math toward SmolLM3's ~37% end-state.
Mix chosen by micro-annealing ablations (OLMo 3 method).

**Stage 2.5 — agentic mid-train, 50–100B (the differentiator).** daVinci-Dev PR corpus
(68.6B tokens; mid-training took non-coder Qwen2.5 bases to 56–58% SWE-bench-V, arXiv:2601.18418)
+ Toucan-1.5M MCP trajectories (arXiv:2510.01179) + Nemotron-Pretraining-SFT-v1 + **our own
engine's output** (§8). Evidence says long-horizon agency must be seeded before post-training
(Youtu-LLM >200B trajectory tokens; AgentFounder "optimization tensions"; arXiv:2509.13310).

**Stage 3 — long-context, 50–100B.** Curriculum 32k→128k→512k, ABF theta ~1M on
full-attention layers, length-stratified (ProLong data: code repos + books + arXiv are the
only real >64k sources) + **Qwen2.5-1M-style synthetic long-range tasks** (FIM, keyword
retrieval, paragraph reordering) — mandatory for Indic (almost no native >64k Indic docs).

**Anneal/decay** coincides with the best data (instruction-formatted + reasoning) + optional
decay-phase logit KD (§9).

**Tooling**: Ai2 Rust stack (duplodocus/datamap-rs/decon — our planned Rust migration,
already built, Apache-2.0) + datatrove + NeMo-Curator GPU MinHash on the clusters.
DCLM-fastText + FineWeb-Edu + Nemotron ensemble classifiers for our own crawl additions only.
**Decontaminate every stage against the FULL eval suite (incl. agentic + Indic evals) before
the first corpus build — retroactively impossible.** Gopher repetition filters + byte-entropy
floors in the pipeline day one (the D9/step-718 lesson).

---

## 6. Pretraining recipe

### 6.1 Schedule and hygiene
WSD, warmup 2000 steps, **linear decay-to-zero** over final ~10–15% (Bergsma/Cerebras;
lets us extend the stable phase opportunistically toward 10T). Global batch 4M → ramp 8–16M.
WD 0.1, grad-clip 1.0, z-loss 1e-4. Peak LR: **set by the new wind tunnel** (the 1B-era
3e-4 does NOT transfer to a new architecture/width).

### 6.2 Optimizer — MuonClip configuration
- Muon on hidden matrices (linear-block + attention + FFN weights); AdamW (fp32 moments) on
  embeddings, norms, sinks/gates scalars, MTP head.
- **Scaling convention (settled by follow-up research)**: optax/PyTorch spectral factor
  `sqrt(max(1, fan_out/fan_in))` inside the update, **no extra width multiplier on top**
  (the old `muon_disable_mup_scale=true` was correct — it removed a double-scaling bug),
  and **NO K2-style `consistent_rms=0.2` when using a muP-transferred LR** — RMS matching and
  width transfer are mutually exclusive conventions (arXiv:2512.05620 App. G).
- QK-Clip τ=100 always on (K2: benign at 15.5T tokens); telemetry: per-layer update spectral
  norms, weight-norm growth, max attention logits (watchdog patterns port as torchtitan hooks).
- Planning number: Muon buys **1.1–1.4×** over well-tuned AdamW (Stanford/Marin), top of range
  requires correct width scaling. Optional wind-tunnel arm: NorMuon; SOAP check at our
  ~40× overtraining ratio (Stanford-I found SOAP edges Muon there — cheap to test at proxy).

### 6.3 Production precedents de-risking this
Moonlight 16B/5.7T; **Kimi K2 1.04T/15.5T zero spikes**; GLM-4.5 355B/23T;
**Motif-2-12.7B dense/5.5T in FP8 under FSDP** (closest analogue; arXiv:2511.07464).

---

## 7. Wind tunnel (Phase 1 — the architecture is not locked until this passes)

250M → 1B proxy ladder, ≥3 seeds per config (treat deltas below seed spread as zero —
the noise-floor discipline of arXiv:2605.20798). muP-transferred LRs; derive and publish the
**muP-for-hybrid transfer recipe** (LR sweep flat across ≥3 widths under the final convention;
negative-control arm with `consistent_rms=0.2` expected to show LR drift; independent-WD
1/width arm judged on post-decay val loss).

**Candidates and pre-registered kill criteria**:
| Candidate | Gate (at 1B unless noted) | Kill → fallback |
|---|---|---|
| 3:1 hybrid ratio (vs 5:1, vs full-attn control) | MQAR/associative-recall probes + 32k-RULER-mini within noise of control; **post-SFT multi-hop QA >32k not regressed** (the MiniMax failure mode, checked after a light SFT, not just pretrain loss) | SWA 5:1 (128–1024 window) + trained sinks (MiMo-V2/gpt-oss template — same ~6× KV savings, zero kernel risk, less novelty) |
| Our gated-delta variant vs vanilla GDN | ≥ GDN on probes + downstream mini-suite | ship vanilla GDN (still hybrid IP at layout level) |
| MTP head | ≥70% 2nd-token acceptance on agentic-style traces | drop head (pure loss, no entanglement) |
| Gated attention + sinks | training stability at 1.5× LR; no downstream regression | plain attention + QK-norm |
| Engram memory bank (25% split) | ≥+2 pts closed-book QA at 1B AND <5% serving penalty in vLLM prototype | defer to v2 (it is the boldest bet; fine to ship v1 without) |
| SuperBPE lottery ticket | ≥+2 pts avg downstream at 250M, no code-fertility regression | plain BPE |
| mHC (optional) | stable at 1.5× LR + >noise gain | drop |
| **Long-context rehearsal (mandatory)** | at 1B: full 4k→32k→131k→262k extension; monotone RULER/NIAH curves; CP rehearsal | if hybrid fails here → fallback architecture, keep ship date |

Budget: ~2–3 months wall-clock overlapping Phase 0; GPU cost is proxy-scale (~1–2% of main run).

---

## 8. Post-training + the agentic data engine

### 8.1 The data engine (Rank-1 IP; build starts Phase 0, runs continuously)
K2-style synthesis: harvest real MCP tools (3k+ public) → evolve synthetic tools (20k+) →
generate agents/tasks/rubrics → multi-turn trajectories in simulated + real-execution
environments → rubric/LLM-judge rejection sampling. **Our extensions (the actual IP)**:
- **Indic + Indian-SaaS/Gov tool environments** (Tally/GSTN-adjacent/ONDC MCP servers) —
  trajectory data nobody else has, compounds with positioning.
- **Tool-error-recovery objective**: perturbed traces (injected failures + recoveries) —
  currently scaffold-engineered everywhere, trained-in nowhere. Publishable.
- **Context-compaction/folding objective**: train the model to natively manage/compact its own
  250k window and detect degraded memory — open territory, serving-neutral.
- **Horizon curriculum**: macro-actions + subgoal decomposition (arXiv:2605.02572) targeting
  the long-horizon collapse mode (agents at 40–50% short-horizon fall <10% in long histories).

### 8.2 Pipeline (recipes copied from OLMo 3 / Nemotron / Qwen3, all published)
1. **SFT** (~2–3M samples, dual `/think`–`/no_think` template): Dolci (ODC-BY) + xLAM-60k
   (CC-BY-4.0 ⚠️"research" note — legal read) + ToolACE + Hermes-FC (Apache) + SWE-Zero 318k +
   SWE-rebench 67k OpenHands trajectories (CC-BY-4.0) + OpenThoughts3-1.2M (Apache) + engine
   output. **Exclude AM-DeepSeek-R1 (CC-BY-NC).** Hermes tool format + `<think>` tags.
2. **DPO/APO** (~150–250k pairs): tool-verified preferences (WorkBench-style) +
   **SecAlign++-style prompt-injection-resistance preference pairs** (AgentDojo attack success
   14.1→2.1% precedent).
3. **On-policy distillation** (replaces most RL): reverse-KL GKD against a big permissive
   teacher — Qwen3 built its 8B this way at **1/10 the GPU-hours of the full pipeline**;
   Thinking Machines measured 50–100× compute advantage vs RL. Cross-tokenizer is fine now
   (GOLD in TRL; ALM near-lossless) — we keep our own tokenizer.
4. **RLVR polish** (modified GRPO — copy OlmoRL or Nemotron configs verbatim): verl or SkyRL;
   environments **assembled, not built**: SWE-rebench 7,500 pre-built Docker images, R2E-Gym
   8.1k tasks, tau2 Gymnasium env, Harbor for terminal, SandboxFusion for code rewards;
   train inside the OpenHands scaffold we're scored on; K8s from day one (DeepSWE's Docker
   crash lesson); ~16–32 CPU cores per training GPU or rent sandboxes ($2–8k/campaign);
   compact filtering + hidden tests + test-file-edit blocks against reward hacking.
5. **Safety**: Tulu-3 safety mix (CoCoNot/WildJailbreak/WildGuardMix) in SFT+DPO;
   gpt-oss-safeguard-20b (Apache) as runtime policy guard driven by our safety_policy.md;
   deterministic out-of-model tool mediation (allowlists, sandboxing).

**Honest calendar: 3–5 months** for the full stage (environment integration 2–4 wks, reward
hardening 2–4 wks, RL campaigns 4–8 wks, tool/terminal domains 3–6 wks) — overlapped with
pretraining by starting the engine and environments in Phase 0. GPU cost is the cheap part
(~1,000–2,500 B200-hr).

### 8.3 Teachers (licenses verified 2026-07-23)
| Teacher | License | Note |
|---|---|---|
| DeepSeek-V4-Pro/Flash | MIT + explicit "distillation for training other LLMs" permission | cheapest frontier API |
| Qwen3.5-397B / Qwen3.6 | Apache 2.0 | strongest permissive family |
| GLM-4.6/5.x | MIT | top open-weight intelligence index |
| gpt-oss-120b | Apache 2.0 | |
| **Gemma 4** | **Apache 2.0** (first OSI Gemma, 2026-04-02) | Gemma 3 terms obsolete |
| Kimi K2.x | Modified MIT (attribution >100M MAU/$20M-mo — moot at our scale) | trajectory diversity |
| **Avoid**: Llama 4 (forced "Llama-" naming on derivatives), MiniMax M3 (new restrictive license), AM-R1 dataset (NC) | | |

---

## 9. Optional decay-phase logit KD
The old Arrow top-K teacher-cache format + code reuses as-is. Offline top-K=64 caches from a
permissive teacher over the final 100–300B tokens ≈ 680–2,040 B200-hr + ~32 TB/100B tokens —
near-zero marginal cost on owned clusters. MobileLLM-R1/Minitron pattern. Do NOT run
full-pretraining KD (Apple distillation scaling laws: we're past the crossover; ~doubles step cost).

---

## 10. Evaluation program

**In-loop**: per-source val PPL (ported hook), OlmoBaseEval-style easy suite every ~1k steps,
full base suite (MMLU/-Pro, GPQA-D, BBEH, MATH-500, GSM8K, HumanEval+/MBPP+) at major
checkpoints via lm-eval-harness/OLMES + evalchemy post-train.

**Long-context gates** (each extension stage): RULER, HELMET, NoLiMa (non-lexical), LongBench v2,
fiction.liveBench-style; plus **LongFuncEval-style long-context tool-calling** (degradation
7–91% industry-wide — our headline metric).

**Agentic**: BFCL v4, tau2-bench, IFEval/IFBench, Terminal-Bench 2.0, SWE-bench Verified
(OpenHands scaffold), AgentDojo (injection robustness).

**12-month targets (honest, anchored to verified 7–9B scores)**:
| Metric | Target | Anchor |
|---|---|---|
| BFCL v3 | ≥60 | Qwen3-8B 66.3; OLMo-3 49.8 |
| tau2-bench avg | ≥45 | Nemotron-3-Nano 49.0 |
| IFEval | ≥85 | Qwen3-8B 89.4 |
| AIME-25 | ≥60 | Qwen3-8B 69.3 |
| LiveCodeBench | ≥50 | Qwen3-8B 59.5 |
| Terminal-Bench 2.0 | ≥10–15 | fine-tuned Qwen3-8B ≈ 11.8 |
| SWE-bench Verified (OpenHands) | legacy comparability number ONLY — ⚠️ OpenAI deprecated it 2026-02-23 (audit: 59.4% of sampled tasks have flawed tests + contamination; its successor SWE-bench Pro was itself retracted Jul 2026). Primary coding-agent gate = **private rolling-repo suite** (§16) | openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/ (verified real) |
| RULER@256k | ≥85 | Kimi-Linear 84.3@128k |
| NoLiMa retention @128k | ≥85% of short baseline | industry: <50% by 32k |
| **BFCL-Hi** | **>62.4 (beat Gemma-3-27b)** | category headline: "7B beats 27B at Hindi tool use" |
| Long-context agentic (ours) | >60% of short-context success at 250k | industry bar: 7% at 64k |

---

## 11. Compute, cost, timeline

**Main-run math** (C = 6·N·D; B200 dense BF16 2.25 PF; B300 = same FLOPs, +50% HBM — use
B300 nodes for the ≥128k phases, otherwise no premium):
| Tokens | GPU-hours @45%/35% MFU | Days on 64 B200 |
|---|---|---|
| 4T | 46k / 59k | 30 / 39 |
| **6T** | **69k / 89k** | **45 / 58** |
| 10T (stable-phase extension) | 115k / 148k | 75 / 96 |

Add ~10–15% for ablations/restarts/mid-train/long-context. MXFP8 lands runs at the 45% row
or better. (If renting instead of owned: ≈$407–636k at mid-2026 on-demand rates; owned
clusters make this opportunity cost.) Post-training GPU ≈ 1–2.5k B200-hr + sandbox fleet.
Checkpoint storage ~6 TB. Dataloader ~6 MB/s.

**Timeline (overlapping phases)**:
| Phase | When | What |
|---|---|---|
| 0 Foundation | M0–M2 | IP-ledger freeze from old repo; new repo; torchtitan bring-up + Muon port + single-node bake-off; tokenizer v3 + gates; corpus acquisition + dedup/decontam (agentic+Indic evals in the index FIRST); data engine v0; governance carry-over |
| 1 Wind tunnel | M1–M3 | §7 ladder; architecture lock; muP-for-hybrid recipe |
| 2 Main pretrain | M3–M5.5 | 5–6T @ 8k; scaling-law envelope kill-switch; watchdog |
| 3 Mid-train + long-context | M5–M6.5 | Stage 2 + 2.5 + 32k→512k extension; RULER/NoLiMa gates |
| 4 Post-training | M4–M9 (overlaps) | engine at full speed; SFT→DPO→on-policy distill→RLVR→safety |
| 5 Release | M9–M12 | HF modeling code, vLLM/SGLang day-1 (contribute hybrid kernels early — llama.cpp WILL lag ~2.5 months for hybrids; accepted), FP8/NVFP4/AWQ/GGUF ladder, tech report, eval cards, EU Art.53 artifacts, license, launch demos |

**Kill-switches**: wind-tunnel gates (§7); step-1000-class trajectory checks vs proxy
scaling-law envelope at main-run start; optimizer fallback AdamW; architecture fallback
SWA+sinks; per-stage decontaminated eval gates before proceeding.

---

## 12. Governance, compliance, safety posture

- **EU AI Act**: 7B × 6–15T ≈ 2.5–6.3×10²³ FLOP → GPAI baseline (Art. 53) only if placed on
  EU market; far below 10²⁵ systemic-risk. Enforcement from **2026-08-02**. Duties: technical
  docs, copyright policy, public training-content summary (official template). Our data card
  auto-render pipeline maps directly onto the template.
- **India**: DPDP + MeitY AI governance guidelines; prefer Sangraha/HPLT over CulturaX for a
  clean posture. IndiaAI Mission funding worth pursuing (⚠️ eligibility/IP-sharing terms
  unverified), never a dependency.
- Carry over wholesale: model/data cards, license register, ISO 42001 scaffolding,
  render-check CI gate, verify-before-locking rule. New (now mandatory for an agentic release):
  eval card, risk card, incident response, OWASP Agentic Top-10 matrix, AgentDojo scores in
  the model card. ISO 42001 certification path = differentiation precedent (IBM Granite).

---

## 13. Salvage manifest (from the 1B repo)

**Reuse as-is**: data-pipeline concepts + code (4.9k LOC, framework-free), tokenizer training
pipeline, teacher-cache format + code, watchdog/quarantine modules (port as torchtitan hooks),
R2 tooling + bucket layout, governance docs, decontamination index builder (extend benchmark
set first), ~half the test suite, gotchas ledger (§3 of SESSION_HANDOFF — port into new
repo's founding docs verbatim), license-vetting framework.
**Dies**: Keras-3 model/training code, Orbax checkpointing, optax Muon wiring (finding
transfers, code doesn't), RunPod orchestration (owned clusters), Rust migration plan
(Ai2 shipped it), all 1B configs/HPs/checkpoints (archive), 5B pilot corpus.
Estimated salvage: ~95–150 person-days.

---

## 14. Decision register (need owner sign-off; recommendations marked)

| # | Decision | Recommendation |
|---|---|---|
| D1 | Model/family name + trademark check | working codename SAMA-7B; decide before tech report |
| D2 | Indic share: ~12–15% Tier-1 concentrated (sovereign identity load-bearing) vs ~5% light-multilingual | ~~12–15% Tier-1~~ **REVISED per master-plan v2 (§16.4): evidence-ledger-gated.** 12–15% of 6T = 720–900B Indic tokens vs a cleanly-licensed unique pool of ~150–350B → 2–6 effective epochs. Build the per-language ledger (unique tokens, epochs, diversity, fertility) + get product evidence, THEN set the share. The wedge intent stands; the percentage is earned, not declared |
| D3 | License: revenue-gated open weights (Liquid-style) vs Apache vs closed/API | **revenue-gated open** |
| D4 | Token budget: 6T vs extend to 10T | **start 6T under WSD; extend if curves justify** (decision deferred by design) |
| D5 | Engram memory bank in v1 | **wind-tunnel gate decides**; fine to defer to v2 |
| D6 | Team: hire 1–2 (data-engine + RL-infra) vs stay solo | **hire** — and see master-plan v2's three priced tiers (§16.4): solo+2 = standard-hybrid only, 18–24 months; 12–18 = standard hybrid + proprietary data/runtime/objectives; 24–35 = full Page Memory program. G0 decides scope before architecture |
| D7 | EU market placement at launch (triggers Art. 53 artifacts) | prepare artifacts regardless (cheap insurance) |
| D8 | Kimi K3 weights land 2026-07-27 with KDA at 1M | re-check before locking the linear-block design — if KDA ships fully open + kernels, building on/extending KDA beats inventing an alternative |

---

## 15. Top risks (ranked)

1. **Hybrid multi-hop regression at scale** (MiniMax mode) — mitigated by post-SFT wind-tunnel
   gate + pre-registered fallback. The 7B-dense/250k/hybrid combination has no public existence
   proof; our wind tunnel is load-bearing.
2. **Solo-lead bandwidth** — the RL environment fleet + data engine are multi-month builds even
   from parts (D6).
3. **Window closure** — Sarvam/BharatGen close the Indic gap or Qwen refreshes the 8B class to
   262k native. 9–12 month clock.
4. **muP-for-hybrid transfer fails to derive cleanly** — no public recipe; budget wind-tunnel
   time; worst case, tune at 1–2B rungs and scale conservatively (Kimi/K2 did exactly this).
5. **MXFP8 parity drift over 6T** — BF16 fallback budget (+~30% time) held in reserve.
6. **License/legal**: Nemotron Data Agreement, xLAM research-note, Toucan license,
   Nemotron-v3 per-subset audit — legal pass in Phase 0.
7. **llama.cpp/edge lag for hybrid** — accepted for v1 (vLLM/SGLang day-1 is the bar);
   mitigate by contributing kernels or shipping a distilled SWA variant later.

---

*Full research corpus (12 dimension reports + 4 follow-ups + verification verdicts + repo
audit) is preserved in [docs/research/7b_pivot/](research/7b_pivot/).*

---

## 16. Addendum (2026-07-23) — cross-review with `7B_AGENTIC_262K_MASTER_PLAN_2026-07-23.md`

A second, independently authored master plan was reviewed and **fact-checked** (code audit
reproduced locally; every load-bearing citation verified against live sources by a 4-agent
adversarial pass). Verdict: **high-quality document; citations check out; treat as a serious
sibling plan.** That doc is left unmodified; this section records what changes here as a result.

### 16.1 Verified findings that CORRECT this plan (adopted above)

1. **Throughput-accounting bug in the legacy stack — CONFIRMED in code.**
   `scripts/benchmark_throughput.py:386` computes `tokens_per_step = micro_batch × seq_len ×
   n_devices`, but the batch from `batch_pairs` is fed to the jitted `train_step` under
   `data_sharding` (line 347) — it is the **global** batch, split across devices. All
   multi-GPU throughput/MFU numbers from that script are inflated ≈ world size: legacy C2
   "30%/46% MFU on 4×B200" was really **~7.5%/11.5%**. Consequences: (a) never reuse legacy
   C2/C3-derived economics; (b) "distributed throughput counts global tokens exactly once"
   is a Phase-0 reliability gate with a unit test; (c) added to the gotchas ledger.
2. **OlmPool (Ai2, 2026-04-23, verified real)** — 26 controlled 7B models: QK-norm is the
   *largest single* long-context degrader (removing it = +6 HELMET@32k); SWA added to a GQA
   model −9 pts; ≥3 stacked choices up to −47%; short-context proxies failed to predict
   32k/64k quality. → QK-norm/GQA-ratio/SWA/norm-ordering are now **pre-registered
   long-context ablations** in the wind tunnel, not frozen defaults, and the wind tunnel's
   gate metrics must include 32k/64k HELMET-class scores from the first rung. Working
   hypothesis: our MuonClip QK-Clip provides logit-explosion safety without always-on
   QK-norm.
3. **SWE-bench Verified deprecated by OpenAI (2026-02-23, verified real)** — 59.4% of audited
   tasks have flawed tests + universal contamination; successor SWE-bench Pro itself retracted
   Jul 2026. → primary coding-agent release gate becomes a **private rolling-repository
   suite** (fresh tasks from live repos, decontaminated by construction); public legacy
   scores reported for comparability only. This *strengthens* the build-our-own-evals IP line.
4. **NVIDIA 26,006 tok/s/GPU (FP8 Llama-3-8B @8k, DGX B200, NeMo 25.02 — verified)** adopted
   as the official planning **ceiling** for 7B-class throughput (~55% BF16-basis MFU).
5. **Ai2 hybrid-token-prediction analysis (verified)** — recurrence wins on evolving semantic
   state, attention on verbatim lookup — added as mechanistic support for the hybrid design
   and for wind-tunnel probes that test *both* pathways (MQAR + verbatim-copy).

### 16.2 Adopted from the master plan (compatible upgrades)

- **Two-lane controls**: every proprietary candidate compared at matched data/FLOPs against
  (a) full-attention Transformer, (b) local/global Transformer, (c) public-style 3:1
  GDN hybrid, (d) recurrent+local without directory. (Upgrades §7's control discipline.)
- **262K certification contract** (release may claim 262k only if): ≥85% retention of the 32k
  registered-suite aggregate at 262k; >95% exact evidence retrieval across position bins
  (incl. near token 250k); full-context reasoning measured, not needle-inferred; TTFT/
  throughput/cache/concurrency published internally for 1/4/8 concurrent requests.
- **Nested recall budgets** (low/balanced/exact context-recall control, separate from
  thinking budget) — added as a wind-tunnel candidate; product-level latency/quality knob.
- **Provenance-gated memory banks** (authority / evidence / verified-state, runtime-assigned)
  — added as a wind-tunnel candidate with their gates (≥50% lower adaptive-injection success
  vs role tags at ≤2 pts benign cost). Backed by verified prior work: role-confusion
  (arXiv:2603.12277, ICML 2026), inseparability theorem (arXiv:2606.27567), CaMeL
  (arXiv:2503.18813). Defense-in-depth; the deterministic broker stays the boundary.
- **Intent-Bound Action Certificate + authority broker + predicted-vs-attested state delta**
  — adopted into the runtime plan (§8/§12 of the master plan); aligns with the verified NIST
  NCCoE agent-identity concept paper (Feb 2026).
- **Workflow Genome quantified targets** for our data engine: 25–40 environment families,
  ≥10k parameterized task families, ~1M verified trajectories, ≥20% held out **by generator
  family**; counterfactual minimum-authority reward as a publishable training-method IP bet.
- **IP hygiene package**: dated invention records, AI-SBOM, no tokenizer/weight transplant,
  signed training ancestry, need-to-know repos, **file-before-disclosure** review, India
  CRI-2025 patent mapping, patentability vs FTO run separately.
- **Ops details**: pinned container digests (no nightly during paid runs); B200/B300 as
  separate homogeneous pools; 30–50% short-context replay during extension (until ablation
  sets the minimum); positional-encoding ablation menu (incl. page-index factorization and
  log-distance buckets) instead of frozen large-theta RoPE; ≥50% of max-length examples must
  be coherent single artifacts, not packed shorts.
- **Legacy repo action**: tag/freeze this repo as the pilot lineage; new private repo for the
  7B program with fresh dependency ledger + signed ancestry.

### 16.3 Where this plan pushes back / open reconciliation (needs owner decision — extends §14)

| # | Tension | Position |
|---|---|---|
| D9 | **Team size**: master plan assumes 24–35 people + 256–1,024 GPUs; this plan was scoped solo+2 | The master plan's scope (Structure-Aware Page Memory + 7 custom kernels incl. CP-aware page gather + certificate runtime + 25–40 env families) is **not executable solo** — it is coherent only with its staffing plan. Decide the company size first; the architecture ambition follows from it, not vice versa. |
| D10 | **Architecture lane**: their Page Memory (R/L/D + 512-token pages + directory) vs this plan's GDN-variant 3:1 hybrid | Not mutually exclusive: the 3:1 hybrid **is** their fallback ("best standard hybrid + proprietary data/runtime moat"). If staffed ≥10 engineers: run Page Memory as Lane B-2 through the 350M/1.3B gates alongside the hybrid. If lean: hybrid is Lane B, Page Memory deferred; its prior-art matrix (Landmark Attention, NSA — both verified) is close, so its patent case needs counsel either way. |
| D11 | **Framework**: their Megatron Bridge (pinned 26.06) vs this plan's torchtitan | Converge via the Phase-0 single-node bake-off both plans already require. Bridge = vendor-supported CP at 262k + the verified 26k tok/s recipe; torchtitan = hackability for custom blocks. Either satisfies both docs. |
| D12 | Token budget ceiling: their 8T planned/10T contingency ≈ this plan's 6T floor/10T extension | No real conflict — adopt "6T committed / 8T planned / 10T contingency" phrasing. |

### 16.4 Master plan **v2** reconciliation (2026-07-23, later same day)

The master plan was revised to v2 (785 → 1,101 lines) after "independent architecture, systems,
data-rights, and execution review." v2 **converges with this plan** on the big calls and
**supersedes parts of §16.3**. Status:

**What v2 absorbed (now shared ground):** four-lane experiment rule whose Lane 2 (public 3:1
GDN hybrid) IS this plan's architecture and whose Lane 4 (public hybrid + one company objective)
IS this plan's training-method-IP bet; the identical compute table (69k/89k GPU-hr for 7B×6T,
extended to 8B/8T rows + 1.3–1.8× program reserve); 6T committed / 8T planned / 10T contingency
(closes D12); torchtitan acknowledged via **Gate S0** same-model bake-off (closes D11 —
though see below); three priced staffing tiers with **G0 scope gate before architecture**
(closes D9: solo+2 = standard-hybrid-only at 18–24 months; 12–18 = standard hybrid + proprietary
data/runtime/objectives/evals; 24–35 = full Page Memory); solo-viable path explicitly defined.

**v2 pushbacks this plan ACCEPTS (positions revised):**
1. *Architecture naming ≠ IP.* Public techniques (3:1 layout, GDN/KDA-class recurrence, gating,
   sinks, partial RoPE, MTP, custom `model_type`) cannot be bundled and branded as original.
   The "named block" stays an internal codename, not an IP claim. Originality claims restrict to
   isolated-evidence mechanisms, objectives, data, environments, runtime, kernels, evals.
2. *Indic share is ledger-gated, not declared* (D2 revised above — my own pool numbers imply
   2–6 effective epochs at 12–15%; the per-language unique-token/epoch ledger decides).
3. *Optimizer rigor*: AdamW is the mandatory control; Muon/MuonClip are candidates. Run the
   **QK-norm × QK-Clip 2×2 factorial** (my "QK-Clip replaces QK-norm" was a hypothesis — this is
   its proper test). Tune WD/τ rather than importing 0.1/100. Note: QK-Clip covers only global
   softmax-attention Q/K (not recurrent blocks) and needs correct global per-head reductions
   under CP.
4. *On-policy distillation is gated, not assumed*, and — the operational catch — token-level
   on-policy/logit KD needs **reproducible logprobs for arbitrary student continuations**, which
   chat/top-K APIs don't provide → **self-hosted teachers on our clusters** for that stage
   (API teachers remain fine for sequence-level trace SFT).
5. *SWA is an experimental arm, not the automatic fallback* — failed candidates return to the
   strongest **measured** control (dense or public hybrid). Consistent with OlmPool's SWA+GQA
   compounding.
6. *Corpus admission contract* (per-content rights, signed token ledger, removal lineage)
   replaces this plan's looser "Dolma is ODC-BY — safe" shorthand: dataset-level labels are not
   per-content clearance.
7. Precision upgrades adopted: 262,144 = prompt+generation budget (246,144 + 16k default);
   5-bucket ≥500-example certification with ≤1-pt short-context regression; complete tensor
   census ("7B means the whole model incl. embeddings"); tied-vs-untied embeddings measured,
   not defaulted; teacher-cache storage corrected to **38.4 TB/100B tokens** (repo's actual
   bf16-logit + uint32-index format).

**Framework recommendation FLIPPED**: Megatron Bridge as default, torchtitan as the four-week
S0 challenger. Decisive verified fact: **Megatron Core ships a native GatedDeltaNet operator
with backward passes and context-parallel (a2a cp↔hp) support**
(docs.nvidia.com/megatron-core/.../core.ssm.gated_delta_net.html — fetched 2026-07-23), which
closes this plan's open question on linear-layer sequence parallelism. One S0 design flag: v2
requires "AdamW plus distributed-Muon correctness" in both harnesses, but torchtitan has no
merged distributed Muon (verified §16.1-era finding) while Megatron has the Moonshot lineage —
either scope S0's optimizer criterion per-harness or accept that it hands Bridge a head start.

**Still ours / complementary (v2 doesn't cover):** the verified market/positioning layer
(competitive-field gaps, BFCL-Hi 62.42 target, monetization comps), the eval anchor table
(§10), the teacher/license tier table and dataset inventory (§8.3, §5), and the research corpus
in `docs/research/7b_pivot/`. v2's counterpart asks: 10 design-partner interviews, 3 prototypes,
2 written pilot commitments before locking the wedge — adopt that as the validation mechanism
for our positioning claims.

### 16.5 Citation-precision nits found in the master plan v1 (for the record; doc left unchanged)

- Qwen2.5-1M's 4k→32k→64k→131k→262k ladder is in the technical report (arXiv:2501.15383 §3),
  not the cited blog (blog says only "4K to 256K gradually").
- CaMeL's paper title is "Defeating Prompt Injections by Design"; CaMeL is the system name.
- OlmPool tested GQA as present/absent, not "aggressive GQA" ratios specifically.
- Its SWE-bench note should not point to SWE-bench Pro as successor (Pro retracted Jul 2026).
