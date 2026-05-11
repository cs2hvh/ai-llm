# AI Research Dossier — MyLLM Audit (2026-05-11)

**Author:** Research role, MyLLM project
**Owner:** harshit.hv@samatva.com
**Status:** Authoritative reference, valid 6-12 months pending major SOTA shift
**Scope:** Decoder-only English-primary 1B foundation model, $15M budget ceiling, Path B + sovereign hedges positioning per `playbook_alignment.md`

---

## Section 1 — 2025-2026 state of the art for small open-weight LLMs (1-3B parameter class)

The 1-3B class has matured into a genuinely competitive product segment over the last 18 months. Three cohorts dominate: Western frontier labs that ship a "lite" variant alongside their main release (Llama 3.2 1B/3B [1], Gemma 3 1B [2], Phi-4-mini [3], Ministral 3 3B [4]); Asian open-weight labs that target the size class as a primary product (Qwen 3 0.6B/1.7B [5], MiniCPM 3 4B [6], DeepSeek-distilled smalls [7]); and academic / corporate-research releases optimised for full reproducibility (SmolLM2 1.7B and SmolLM3 3B [8][9], OLMo 2 1B [10], StableLM-2 1.6B [11], IBM Granite 3.0/3.1/4.0 small [12][13], Sarvam-1 2B [14]).

### 1.1 The current SOTA spec sheet (verified configs)

| Model | L | Hidden | FFN | FFN/h | Q heads | KV heads | head_d | Vocab | Ctx native | Train tok | RoPE θ | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Llama 3.2 1B [1] | 16 | 2048 | 8192 | 4× | 32 | 8 | 64 | 128k | 8k → 128k | ~9T | 500k | scaled init, tied |
| Llama 3.2 3B [1] | 28 | 3072 | 8192 | 2.67× | 24 | 8 | 128 | 128k | 8k → 128k | ~9T | 500k | distilled from 8B/70B logits |
| SmolLM2 1.7B [8] | 24 | 2048 | 8192 | 4× | 32 | 32 | 64 | 49k | 8k | 11T | 130k | NO GQA; WSD schedule |
| SmolLM3 3B [9] | 36 | 2048 | 11008 | 5.38× | 16 | 4 | 128 | 128k | 64k → 128k YARN | 11.2T | mixed RoPE/NoPE | every 4th layer NoPE; 3-stage curriculum |
| Qwen 3 0.6B [5] | 28 | 1024 | 3072 | 3× | 16 | 8 | 64 | 151k | 32k | undisclosed multi-T | 1M | tied, GQA 2:1 |
| Qwen 3 1.7B [5] | 28 | 2048 | 6144 | 3× | 16 | 8 | 128 | 151k | 32k | undisclosed multi-T | 1M | tied; off-policy distill |
| Gemma 3 1B [2] | 26 | 1152 | 6912 | 6× | 4 | 1 | 256 | 262k | 32k | 2T | 1M | local-global 5:1 SWA; text-only |
| Phi-4-mini 3.8B [3] | 32 | 3072 | 8192 | 2.67× | 24 | 8 | 128 | 200k | 128k | undisclosed | 250k | tied embeddings (new for Phi-4) |
| Ministral 3 3B [4] | 26 | — | — | — | — | — | — | — | 128k | distillation from 14B | — | Apache-2.0; pruning + distillation recipe |
| OLMo 2 1B [10] | 16 | 2048 | 8192 | 4× | 16 | 16 | 128 | 100k | 4k | 4T (3T + 1T anneal mix "Dolmino") | 500k | fully open; RMSNorm + post-norm in attn |
| Granite 3.0 1B-A400M [12] | 24 | 1024 | ~2048 | sparse MoE | 16 | 8 | 64 | 49k | 4k → 128k | 10T | 10k | MoE 1B/0.4B-active |
| Granite 4.0-H-Micro 3B [13] | hybrid | — | — | — | — | — | — | 100k | 512k | ~15T | n/a | 9:1 Mamba-2:transformer dense |
| Sarvam-1 2B [14] | 28 | 2048 | — | — | 16 | 8 | — | 64k | 8k | 4T (2T Indic + 2T En) | 10k | NeMo trained; Indic-optimised tokenizer |
| StableLM-2 1.6B [11] | 24 | 2048 | 5632 | ~2.75× | 32 | 32 | 64 | 100k | 4k | 2T × 2 epochs | partial-rope (25%) | LayerNorm w/ bias; bias-free FFN |

(Where a field is "undisclosed" the lab has not published the technical report or has redacted the value in the config.json. Sarvam-1's RoPE θ = 10k is unusually low for an 8k native context and is one of the few SOTA deviations.)

### 1.2 Convergent design choices (treat as defaults in 2026)

1. **Pre-norm RMSNorm + SwiGLU + GQA + RoPE.** Universal across all dense releases. The Stable LM 2 LayerNorm experiment was not repeated by any 2025 release.
2. **Tied input/output embeddings** for everything ≤ 4B. Phi-4-mini's switch from untied (Phi-3.5) to tied [3] confirms the convention is now strict.
3. **128k-class vocab.** Llama 3.2 (128k), Phi-4-mini (200k), Gemma 3 (262k), Qwen 3 (151k). The 49k SmolLM2 / Granite 3 vocabs are now visibly outdated even by their authors' own follow-ups (SmolLM3 → 128k, Granite 4 → 100k).
4. **RoPE θ ≥ 130k for 8k+ native context.** Llama 3.2's 500k is the median. Qwen and Phi go to 1M and 250k respectively [3][5].
5. **Scaled init for residual projections** at the 1B+ scale (Llama 3.2 [1]). The Llama-style `std/√(2L)` factor is now considered hygiene, not optional.
6. **WSD or WSD-flavoured schedules** for any team that wants to fork checkpoints mid-run. SmolLM2 [8], SmolLM3 [9], MiniCPM [6], and the Pangu-Pro multi-stage pipeline [15] all use it.
7. **Multi-stage data curriculum with math-and-code-heavy annealing.** SmolLM3's three-phase curriculum [9], MiniCPM's WSD-decay-as-curriculum [6], Yi-Lightning's three-stage train [15], and OLMo 2's Dolmino mid-training mix [10] are all variants of "raise the share of high-quality math/code in the final 10-20% of training."
8. **Token budgets in the 6,500–24,000 tok/param range.** Llama 3.2 1B at 7,250:1, SmolLM2 at 6,500:1, SmolLM3 at 3,700:1, Qwen 2.5 1.5B at ~12,000:1 [Architecture-Review §1]. Below 1,000:1 (your previous v0 target) is now considered a research artifact, not a real model.
9. **Distillation from a larger same-family teacher**. Llama 3.2 1B/3B used logits from Llama 3.1 8B/70B during pretraining [1]; Ministral 3 was pruned from Mistral Small 3.1 then distilled [4]; Gemma 3 was distilled from larger sibling [2]. **At 1B-3B, "distillation from a larger model" is no longer optional for the frontier — but it is optional for our positioning** because we are sovereignty-first, not benchmark-first.

### 1.3 Active debates and post-2025-H2 innovations to be aware of

**(a) Hybrid Mamba-2 / transformer.** IBM Granite 4.0 (Oct 2025) ships a 9:1 Mamba-2:attention dense 3B that claims 70% memory reduction and 2× inference speedup at iso-quality [13]. This is now a real architectural alternative, not a curiosity, but it is a *big* deviation from our locked stack and is not yet proven at 1B-from-scratch outside IBM.

**(b) Hybrid RoPE / NoPE.** SmolLM3 [9] omits RoPE entirely on every 4th layer, citing Cohere's RNoPE paper [16]. The claim: stronger long-context generalisation at no short-context cost. This is reproducible behind a layer-config flag and worth considering as a v1.5 experiment.

**(c) Multi-Token Prediction (MTP).** Initially a frontier-only technique (DeepSeek V3). 2025 work shows MTP is feasible at 410M–2.8B and Google has shipped MTP drafter heads for Gemma 4 to get up to 3× inference speedup at zero quality cost [17]. *Caveat*: MTP does not improve generative tasks for small models below the 1B-3B capability threshold [17] — for us this is post-train-only territory.

**(d) FP8 mixed-precision training.** The Scaling FP8 to Trillion-Token LLMs paper (ICLR 2025) [18] and the FP8-and-Back-Again paper [19] establish that FP8 training is feasible but narrows the hyperparameter window. Critical finding [18]: SwiGLU amplifies outliers, causing late-training divergence that BF16 baselines do not see. The mitigation requires per-tensor FP32 scale-factor maintenance — not a free win. **For a 1B model trained for 1-3T tokens on H100s/B200s, BF16 remains the safe default and FP8 is a P2 optimisation.**

**(e) WSM (Warmup-Stable-Merge).** Tian et al. [20] (Jul 2025) prove WSD-style decay is equivalent to averaging recent checkpoints. WSM reports +3.5% MATH, +2.9% HumanEval, +5.5% MMLU-Pro over vanilla WSD. Implementation is essentially free (you must keep the last N checkpoints already). **This is the single highest-leverage methods-paper of the year for our class of project.**

**(f) FineWeb 2 + Nemotron-CC.** FineWeb 2 (Dec 2024, paper Jun 2025) [21] processes 96 CC dumps for 1000+ languages, ODC-By licensed. Nemotron-CC [22] (Dec 2024) is a 6.3T-token English CC corpus with synthetic rephrasing; reports +5 MMLU vs Llama-3.1-8B-replica at 15T tokens. Both are direct upgrades over FineWeb-Edu + Falcon-RefinedWeb for the high-quality-web slot.

**(g) muP / muTransfer.** Hyperparameter transfer from a 40M proxy to a 6.7B production model at 7% of pretrain compute [23]. MiniCPM uses this as their "model wind tunnel" [6]. **For our pilot-then-base flow, muP is the right principled answer** — we already plan to gate the 1B run on pilot signal; muP makes the pilot's LR/init choices transferable instead of being a separate calibration.

**(h) Document masking in FlashAttention.** FlashMask (ICLR 2025) [24] and FlexAttention/FlashAttention-4 [25] now support proper intra-document masking efficiently. This matters because the standard "pack multiple docs into one 4k seq with reset attention mask" pattern was previously much slower than naïve causal masking; it now isn't. SmolLM3 and Qwen 3 both use it [8][9].

**(i) KV-cache compression / MLA.** DeepSeek's Multi-head Latent Attention [7] is a serving-time win, not a training-time choice. Not relevant at our scale; defer.

**(j) Soft-MoE / fine-grained MoE below 7B.** Granite 3.0's 1B-A400M demonstrates this works [12], but the per-token cost of routing eats most of the gain at our parameter count. Playbook N8 (skip MoE below 30B) remains correct.

---

## Section 2 — Audit of `PLAN.md` against the SOTA in §1

Methodology: for every material choice in `PLAN.md`, `architecture_review.md`, `playbook_alignment.md`, and the four configs read above, I assign one of ✅ / 🟡 / 🔴 / ❓ and anchor the call to a specific peer or paper. Items marked ✅ are kept terse; the rest get real reasoning. The lr_schedule cosine→WSD migration in `base_1b.yaml` is excluded as already-fixed.

### 2.1 Architecture (configs/base_1b.yaml, configs/pilot_250m.yaml)

| Choice | Verdict | Note |
|---|---|---|
| Llama-style pre-norm decoder, RMSNorm, SwiGLU, GQA, RoPE | ✅ | Universal in §1.2 |
| Base 1B = 16L / 2048 / FFN 8192 / 32 heads / 8 KV / head_d 64 | ✅ | Exact Llama 3.2 1B match [1] |
| Pilot 250M = 16L / 768 / FFN 3072 / 12 heads / 4 KV / head_d 64 | ✅ | Internally consistent, 4× FFN, 3:1 GQA |
| Vocab 131,072 | ✅ | Power-of-2; safely between Qwen 3 (151k) and Llama 3.2 (128k) |
| Tied embeddings | ✅ | Universal at this scale |
| RoPE θ = 500,000 (base) / 130,000 (pilot) | ✅ | Llama 3.2 / SmolLM2 values respectively |
| Scaled init for residuals (base only) | ✅ | Llama hygiene |
| z-loss 1e-4 | ✅ | PaLM/Llama 2 standard, no peer has dropped it |
| QK-norm off, deferred to "decide post-pilot" | 🟡 | See §3 — recent literature now treats QK-norm as default hygiene at 1B+ [26]; the cost of leaving it off is "wait for divergence then add it" and that costs a re-train |
| `scaled_init_for_residuals: false` on pilot, `true` on base | 🟡 | Inconsistent. Pilot's job is to forecast base behaviour; setting it differently undermines the muP-style transfer intent. |
| Mixed precision: bf16 weights + bf16 activations + fp32 optimiser state | ✅ | Standard; FP8 is P2-only per §1.3(d) |
| **Attention masked-mat-mul implementation** (not document-masked) | 🔴 | See §3 — Pretraining shards are packed multi-doc; without intra-document masking we are training the model to attend across document boundaries. FlashMask/FlexAttention now make this free [24][25]. Architecture-Review F5 already noted this for the perf angle; the *correctness* angle is the real reason. |
| **No muP / muTransfer parameterisation** | 🔴 | The pilot-to-base transition is the most expensive bet in our plan; making it a principled muP transfer reduces re-tuning risk for ~2-engineer-weeks of work [23]. |
| **No MTP** at 1B base | ❓ | Defer to post-training; do not change pretraining |

### 2.2 Optimiser & schedule

| Choice | Verdict | Note |
|---|---|---|
| AdamW β=(0.9, 0.95), wd=0.1, ε=1e-8, grad-clip 1.0 | ✅ | Universal |
| Peak LR 3e-4 (pilot) / 2e-4 (base) | ✅ | Within Llama-3.x band; muP would let us derive this from the pilot |
| 2000-step linear warmup | ✅ | Conservative; could go to 1% of steps |
| WSD with end_lr_ratio=0.1, decay_fraction=0.15 | ✅ | Matches SmolLM2/MiniCPM convention |
| **WSM** (checkpoint-merging-as-decay) | ❓ | New, free, +5.5% MMLU-Pro reported [20]. P1. |
| **Batch ramp 4M → 8M** in base config | ✅ | Standard practice; Llama 3 ramps similarly |

### 2.3 Data (configs/data/pretrain_mix.yaml, configs/tokenizer.yaml)

| Choice | Verdict | Note |
|---|---|---|
| Tokenizer = SentencePiece-Unigram, 131k vocab, byte_fallback, NFKC, metaspace | ✅ | Modern multilingual default; identical to Sarvam-1 architecture choices [14] |
| 7-language plan (en 70% / hi 6% / es-zh-ar-fr 5% each / de 4%) | ✅ | Internally consistent |
| **Tokenizer training corpus draws Hindi from `ai4bharat/sangraha` config "verified"** | ✅ | CC-BY-4.0, commercial-OK [27]. Better than mc4. |
| Pretrain mix: 45% web, 18% code, 6% wiki, 5% books, 6% academic, 7% math (baseline) / 14% (decay), 4% QA, 12% multilingual, 3% structured | 🟡 | Reasonable starting point. Two concerns below. |
| **Web slot = FineWeb-Edu (31.5%) + Falcon-RefinedWeb (13.5%)** | 🔴 | FineWeb-Edu's 90% filtration limits it to ~1.3T unique tokens [22]. For a 1T-token base run this is OK; for 3T-ambitious it is not. Falcon-RefinedWeb is ODC-By [28] but its filtering is 2023-vintage; Nemotron-CC [22] strictly dominates it in 2025 ablations. **Recommend swap RefinedWeb → Nemotron-CC-HQ + bump FineWeb-Edu to FineWeb-Edu-score-2.** |
| **Multilingual = mc4 for es/zh/ar/fr/de + Sangraha for hi** | 🟡 | mc4 is now superseded by FineWeb 2 [21] for 1000+ languages under the same ODC-By license. Switching is mostly mechanical, but mc4's "trust_remote_code: true" loader is currently broken on the HF datasets >= 3.0 line — this is a near-term concern. |
| **Code = bigcode/the-stack-v2** | 🟡 | Opt-out mechanism exists [29] but is publicly contested ("does not honour opt-out requests that have been made in the last year" per the discussion thread on the dataset page). For a commercial-target product we should freeze our dataset revision (specific snapshot date) and document the opt-out reconciliation date in the data card. |
| **Decontamination = n-gram match against HellaSwag/MMLU/GSM8K/HumanEval prompts only** | 🔴 | Insufficient. Modern decon pipelines (FineWeb [21], OLMo 2 [10]) match against the full evaluation suite, including MMLU-Pro, GPQA, IFEval, MGSM, LiveCodeBench, BBH. Skipping LiveCodeBench decon is the most dangerous gap because LiveCodeBench is the only contamination-resistant code bench we have at 1B [30]. |
| **No quality-classifier on web** (just KenLM perplexity + heuristics planned) | 🟡 | FineWeb-Edu's success was driven by an explicit Llama-3-judged quality classifier. We get this for free by *using* FineWeb-Edu (already filtered) but if we add Nemotron-CC we should also adopt its quality ensemble [22]. |
| **No deduplication-against-eval-suite** in tokenizer-training corpus | ❓ | Standard concern; check after tokenizer is trained |
| **No "long-context curriculum" stage** for the 8k→32k YaRN extension | 🔴 | The base config promises "extend later via YaRN" but YaRN extension is *training*, not just config. Llama 3.2 trained the long-context extension on 800B+ tokens at the long sequence length [1]. A pure post-hoc theta-rescale will work for 32k but is the worst-quality option of the three (theta-rescale / YaRN / NTK-aware). At minimum we need a budget line for a long-context anneal pass. |

### 2.4 Training infrastructure

| Choice | Verdict | Note |
|---|---|---|
| Keras 3 + JAX backend | 🟡 | Already locked. Architecture-Review F9 documented the friction. No 2025-26 small-LLM was trained on this stack from scratch (production stacks: MaxText, NeMo, TorchTitan, gpt-neox, internal Meta/Google stacks). Risk is non-zero but the model code is portable. **Hold the line; canary on the pilot.** |
| Optax AdamW + Orbax checkpointing | ✅ | Mature; Orbax sharded ckpts are the JAX standard |
| `jax.numpy.dot_product_attention` not used | 🔴 | Architecture-Review F5 deferred this; with FlashMask now released [24] and `jax.nn.dot_product_attention` supporting GQA + causal masking out of the box [31], the right move is to use it from day one of the pilot. Memory savings at 8k context are 4-16× depending on head config. |
| FSDP via `Mesh(("data","model"))` + `NamedSharding` | ✅ | Standard JAX SPMD pattern |
| 4× pod-of-8 H100 SXM for base 1B run | 🟡 | RunPod multi-node interconnect is the open risk (R1 in PLAN.md). At 1B / 1T tokens, **a single 8× H100 SXM pod for ~6-8 weeks would also work** and removes the multi-node failure mode. Cost is similar; wall-clock is longer. Decide at booking time. |
| B200 single-node for pilot | ✅ | Clean choice; 180 GB HBM is comfortable for 250M at 4k seq + bf16 |
| WandB SaaS tracking | ✅ | Industry standard |
| Cloudflare R2 storage | ✅ | Zero-egress decision is materially good given Orbax checkpoint sizes |

### 2.5 Evaluation strategy (PLAN.md §9)

| Choice | Verdict | Note |
|---|---|---|
| HellaSwag, ARC, MMLU, WinoGrande, TruthfulQA | ✅ | Baseline; saturated at frontier but still informative at 1B |
| MMLU-Pro, BBH, GSM8K, MATH, HumanEval+, MBPP+, LiveCodeBench | ✅ | Strong gate set |
| IFEval, MT-Bench, Arena-Hard | ✅ | Post-SFT |
| XCOPA, XNLI subset, MGSM | 🟡 | OK but MGSM is now mostly replaced by **MMLU-ProX** [32] and **Global-MMLU** [33] for serious multilingual evaluation. Belebele [33] is the de-facto multilingual reading-comp bench. |
| **Long-context evals: NIAH only, planned implicit** | 🔴 | NIAH is now considered insufficient by itself; **RULER** [34], **BABILong** [35], and **NoLiMa** [36] are the 2025 standards. Even models that pass NIAH at 128k fail RULER beyond 10% of their context. For us at 8k native this matters less, but we need at least RULER-4k for the gate. |
| **Agentic tool-use eval: only Glaive/ToolBench for training** | ❓ | No agentic *evaluation* in §9. Need at least τ-bench or BFCL v3. |
| **Safety: HarmBench + AdvBench tier-1, internal probes tier-2** | 🟡 | Should add **JailbreakBench** [37] (de-facto NeurIPS 2024 standard, overlaps but adds reproducible defence comparison). |
| **Refusal calibration metric** | 🟡 | Need an over-refusal (XSTest) bench, not just refusal of harmful |
| **No contamination report on each bench** | 🔴 | Modern releases (OLMo 2, SmolLM3) ship a per-bench contamination % computed against the training set. We should do the same; it's the difference between a credible eval and a sales sheet |

---

## Section 3 — Top prioritised concrete recommendations

Ten items, ordered by impact. I am being conservative about engineer-hour estimates because every estimate balloons in practice.

### **R1 (P0). Adopt muP / muTransfer for pilot-to-base hyperparameter transfer.**
- **What:** Implement muP scaling in `src/myllm/model/layers.py` (init scaling, attention scaling, output scaling) and `src/myllm/training/optim.py` (per-layer LR scaling). Run pilot **and** a 30M-param "model wind tunnel" sweep at the start of Phase 2 to anchor LR + init [23][6].
- **Why:** The single most expensive bet in the project is the 1B base run. Without muP, every HP from the pilot is interpolated by-eye and we'll likely re-tune at 1B, which costs an extra pilot-equivalent. With muP, pilot's optimal LR transfers zero-shot to base.
- **Cost:** 2 engineer-weeks. Compute: ~$2k for 30M sweep at 8 grid points × 1B tokens each.
- **Risk if we don't:** Either we get the 1B LR wrong (loss diverges, costs 1-2 weeks + $10-40k to roll back) or we leave performance on the table from a too-conservative LR.
- **Phase:** Phase 2 prep, before pilot kickoff.
- **Priority:** P0.

### **R2 (P0). Use `jax.nn.dot_product_attention` from day one; intra-document attention masking from day one.**
- **What:** Replace explicit `softmax(QK^T/√d) @ V` in `src/myllm/model/layers.py` with `jax.nn.dot_product_attention(q, k, v, mask=segment_ids[..., None] == segment_ids[..., None, :], is_causal=True)`. Generate per-batch `segment_ids` in the packer.
- **Why:** Without document masking, packed shards train cross-document attention. This is a *correctness* gap, not a perf optimisation — every 2025 SOTA training run (SmolLM3, Qwen 3, Llama 3) uses doc-masked attention via FlashMask or equivalent [9][24]. The fused kernel cuts attention memory 4-16× at 8k context [31].
- **Cost:** 3-5 engineer-days. Free compute-side.
- **Risk if we don't:** Quality degradation that's invisible until eval; perf wall at 8k context training.
- **Phase:** Phase 0 / Phase 2.
- **Priority:** P0.

### **R3 (P0). Enable QK-norm by default at base scale; consider for pilot.**
- **What:** Set `qk_norm: true` in `configs/base_1b.yaml`. Plumb in `src/myllm/model/layers.py` as a pre-attention RMSNorm on Q and K (post the RoPE rotation, per Llama 3 convention).
- **Why:** QK-norm is now the consensus stability primitive at 1B+ [26]. Gemini 2.0, DeepSeek-V3, Llama 3 70B (yes, 70B uses it; Llama 3.2 1B does not but at our recipe it is risk-free) all ship it. Cost is one extra norm per layer (~0.3% FLOPs); benefit is robustness to LR perturbations of 1.5× and elimination of late-training divergence.
- **Cost:** 1 engineer-day.
- **Risk if we don't:** Loss spikes in late base pretrain (Risk R2 in PLAN.md); spike rollback protocol kicks in; lost wall-clock.
- **Phase:** Phase 4.
- **Priority:** P0 for base; P1 for pilot (pilot is shorter so divergence less likely).

### **R4 (P1). Replace Falcon-RefinedWeb with Nemotron-CC-HQ in the web slot.**
- **What:** Edit `configs/data/pretrain_mix.yaml`: swap the `tiiuae/falcon-refinedweb` 13.5% row for `nvidia/Nemotron-CC` (HQ subset only) at the same share. Add Nemotron-CC's synthetic-rephrasing subset as an additional 5% via re-balancing.
- **Why:** Nemotron-CC reports +5.6 MMLU vs DCLM on a held-out 8B-model evaluation [22]. RefinedWeb's 2023-vintage filters are visibly weaker. Both are CC-derived and ODC-By compatible.
- **Cost:** 1 engineer-day config + tokenization re-run on the new slice (~$200).
- **Risk if we don't:** Leave 2-5 points of MMLU on the table at the 1B scale, which is a *lot* for our gate criteria.
- **Phase:** Phase 3 / Phase 4.
- **Priority:** P1.

### **R5 (P1). Adopt WSM (checkpoint-merge-as-decay) on top of WSD.**
- **What:** Modify `src/myllm/training/checkpoint.py` to keep the last N (=10) sharded Orbax checkpoints from the stable phase and, at decay-time, produce both: (a) the WSD-decayed final, and (b) a WSM-merged final via weight averaging. Evaluate both at gates.
- **Why:** +3.5% MATH, +2.9% HumanEval, +5.5% MMLU-Pro reported in [20] for free at small models. Compatible with our WSD schedule. Adds essentially zero compute (the merge is a single weighted average of already-stored checkpoints).
- **Cost:** 2 engineer-days. Storage: 10 × ~5 GB sharded ckpts on R2 ≈ negligible.
- **Risk if we don't:** Leave free-money quality on the table; doesn't block the project.
- **Phase:** Phase 4 decay; Phase 5 continued-pretrain.
- **Priority:** P1.

### **R6 (P1). Tighten the decontamination pipeline; ship a per-bench contamination report at every gate.**
- **What:** Extend `src/myllm/data/decontamination.py` to n-gram match against MMLU-Pro, GPQA, IFEval, MGSM, BBH, LiveCodeBench-Lite, JailbreakBench-JBB and the planned multilingual evals. Emit a CSV at every gate: `{bench, % positives, % matched, max_ngram, sample}`. Adopt the OLMo 2 reporting format [10].
- **Why:** Without this, gate scores aren't credible to enterprise users or to ourselves. Architecture-Review §5 deferred this to Phase 3 — we should treat it as part of Phase 3 not "later."
- **Cost:** 1 engineer-week (the eval set is large but the matching pipeline reuses MinHash/LSH infra).
- **Risk if we don't:** Reputational risk; can't claim "uncontaminated" on releases; EU AI Act transparency obligations [38] expect this.
- **Phase:** Phase 3.
- **Priority:** P1.

### **R7 (P1). Long-context plan: replace "YaRN at end" with a proper long-context anneal.**
- **What:** Add a Phase-4.5 line item: after WSD decay, run an extra 50-100B-token anneal at 32k native context, doc-masked. Update RoPE θ post-decay (do not "rope-rescale" — train with the new θ). Add `configs/longctx_anneal.yaml`.
- **Why:** Llama 3.2 1B's 8k→128k extension cost ~800B tokens of long-context training [1]. We can get to a credible 32k for ~5-10% of base-run cost. Pure post-hoc rope-rescale or YaRN-on-cold-weights gives 32k context but with badly degraded retrieval. RULER scores at 32k are visibly worse without anneal [34].
- **Cost:** 1 engineer-week + ~$20-40k compute (50-100B tokens).
- **Risk if we don't:** Marketed 32k context fails RULER at 16k+; serving applications that depend on long context break.
- **Phase:** Phase 4.5 (new).
- **Priority:** P1.

### **R8 (P1). Multilingual eval upgrade: switch from MGSM/XCOPA-only to MMLU-ProX + Global-MMLU + Belebele + MGSM.**
- **What:** Add to `src/myllm/eval/`: MMLU-ProX (29 langs) [32], Global-MMLU-Lite [33], Belebele (reading comp, 122 langs) [33]. Drop XCOPA from gate set (keep XNLI for sanity). Hindi-specific: add MILU (already listed as supplementary in PLAN.md) and run it at every gate, not just supplementary.
- **Why:** XCOPA is saturated at frontier and stops discriminating at 1B; MMLU-ProX and Global-MMLU are the 2025-26 standards. With 12% multilingual mix and Hindi from Sangraha we need real Hindi evals to know if the spend is paying off.
- **Cost:** 3 engineer-days (lm-eval-harness already supports most; Global-MMLU-Lite needs a wrapper).
- **Risk if we don't:** Multilingual investment is unmeasured; can't make any defensible claim about Hindi capability post-train.
- **Phase:** Phase 12 (eval), pre-emptively wired in Phase 0/1.
- **Priority:** P1.

### **R9 (P2). Document and prepare-for the EU AI Act GPAI obligations even though we are below the 10²⁵ FLOP threshold.**
- **What:** Author `docs/eu_ai_act_disclosure.md` containing: (a) FLOP count estimate for pilot + base (we are at ~3 × 10²² for 1B / 1T tokens — three orders of magnitude below the threshold [38]); (b) summary-of-training-data disclosure per Annex template; (c) copyright opt-out reservation policy (where we honour `robots.txt` and Common Crawl opt-outs, where the-stack-v2 opt-outs apply); (d) downstream-provider information sheet.
- **Why:** From 2 Aug 2025, all GPAI providers must comply with transparency and copyright obligations to place models on the EU market [38]. We are not a "systemic risk" provider but we are a GPAI provider the moment we open-weight release. Enforcement is from 2 Aug 2026 — we will be in the enforcement window when we ship.
- **Cost:** 2 engineer-days + 1 day of legal review (which we don't have in-house; budget for outside counsel).
- **Risk if we don't:** EU distribution requires it; "open" releases are not exempt.
- **Phase:** Phase 0 (start) and Phase 13 (ship).
- **Priority:** P2 (regulatory, not technical).

### **R10 (P2). Confirm-and-document the Teacher API legal posture before Phase 6.**
- **What:** Resolve open item §14.6: the teacher-API model identity must be confirmed before any synthetic SFT/DPO/CoT data is generated. Update `docs/playbook_alignment.md` to spell out: if teacher = DeepSeek-V3.2 → MIT, free use [7]; if teacher = Llama-3.x → output-derived models must be renamed with "Llama-" prefix [39]; if teacher = Gemma → outputs cannot be used to train competing models, and "Model Derivatives" includes synthetic-output-trained models per Gemma's terms [40]; if teacher = GPT-4 or Claude → OpenAI/Anthropic ToS forbid using outputs to train competing models.
- **Why:** Gemma's terms make output-distillation legally toxic for our positioning [40]. Llama's "Llama-" naming requirement applies to *any* derivative including ones trained on Llama outputs [39]. DeepSeek-V3.2-MIT is the only frictionless option [7].
- **Cost:** 1 hour of project-owner decision + 1 engineer-day to document.
- **Risk if we don't:** SFT synthesis runs spawn an irreversible licensing dependency; cannot release model commercially.
- **Phase:** Pre-Phase 6.
- **Priority:** P2 (gate, not technical).

---

## Section 4 — Enterprise concerns

### 4.1 Licensing audit of `configs/data/pretrain_mix.yaml`

I evaluated every dataset in the mix against four axes: license, commercial-use clause, attribution requirement, and provenance / opt-out risk.

| Dataset | Share | Stated license | Commercial OK? | Attribution? | Provenance / opt-out risk |
|---|---|---|---|---|---|
| `HuggingFaceFW/fineweb-edu` [41] | 31.5% | ODC-By 1.0 | Yes | Required | Common Crawl ToU applies; CC honours robots.txt at crawl time but no per-doc opt-out |
| `tiiuae/falcon-refinedweb` [28] | 13.5% | ODC-By 1.0 | Yes | Required | Same as above; 2023-vintage filter |
| `bigcode/the-stack-v2` [29] | 18% | Mix (per-file original license preserved) | Conditional | **Per-file attribution required** | **Contested**: opt-out mechanism exists but is alleged not to honour all recent requests; we must pin a specific snapshot and document the opt-out reconciliation date in the data card |
| `wikimedia/wikipedia` | 6% | CC-BY-SA 4.0 | Yes | Required (ShareAlike does not apply to model weights per current legal consensus, but uncertain) | Low |
| `pg19` [42] | 5% | Apache-2.0 (dataset wrapper) over pre-1919 public-domain books | Yes | Apache attribution | Project Gutenberg trademark — do not use the PG name in product naming |
| `allenai/peS2o` [43] | 6% | ODC-By | Yes | Required | Some underlying papers are non-OA; allenai claims the dataset is OA-only but this is paper-by-paper |
| `open-web-math/open-web-math` | 2.8% | ODC-By 1.0 | Yes | Required | CC-derived; same posture as FineWeb |
| `EleutherAI/proof-pile-2` [44] | 4.2% | Component-licensed (**not uniform**) | Component-by-component | **Per-component attribution; some non-commercial subsets** | **Watch this**: ArXiv subset is OK, but algebraic-stack and OpenWebMath have mixed provenance |
| `HuggingFaceH4/stack-exchange-preferences` | 2% | CC-BY-SA 4.0 | Yes (with ShareAlike caveat) | Required | Low; SE network's standard license |
| `ai4bharat/sangraha` [27] | 4% | CC-BY-4.0 | Yes | Required | Low; explicit |
| `mc4` (es/zh/ar/fr/de) [45] | 6% | ODC-By 1.0 | Yes | Required + CC ToU | Common Crawl ToU; broken loader on newer HF `datasets` versions |

**Bottom line:** all 11 sources are commercially compatible in principle. Three are non-trivial: **the-stack-v2** (per-file licenses, contested opt-out), **proof-pile-2** (component-by-component), and **mc4** (operational rather than legal risk). For an enterprise release we should:
1. Pin the exact dataset snapshot / commit hash in `configs/data/pretrain_mix.yaml` and never silently re-pull.
2. Author a **data card** at Phase 3 close listing, for every source: license, version pin, attribution string, retrieval date, filter summary.
3. Honour the-stack-v2 opt-outs current as of our snapshot date; cite the snapshot date prominently.
4. Replace `mc4` with `HuggingFaceFW/fineweb-2` for the secondary languages [21] — same license, same upstream Common Crawl, better quality, currently maintained.

### 4.2 Teacher-API distillation legal map

This map is the source-of-truth for §14.6 in PLAN.md, replacing the brief note there.

| Teacher | License | Distillation OK? | Output ownership? | Constraint propagation? | Net verdict |
|---|---|---|---|---|---|
| **DeepSeek V3.2 / V3.1** [7] | MIT | Yes, explicitly permits commercial use and distillation | User of outputs | None | **Cleanest option**. No naming, no propagation. |
| **Llama 3.x** [39] | Llama 3.x Community License | Yes, but "derivative models" trained on outputs must be named with "Llama-" prefix and include attribution notice | User | "Llama-" naming requirement on all downstream commercial derivatives | **Toxic for our positioning** — our product would have to be "Llama-MyLLM" or similar. Not acceptable. |
| **Gemma 3 / 4** [40] | Gemma 3: Gemma Terms of Use; Gemma 4: Apache-2.0 (Gemma 4 changes the rules) | Gemma 3: **No** — "Model Derivatives" explicitly includes "models created by transfer of patterns of the weights, parameters, operations, or **Output of Gemma**, to that model in order to cause that model to perform similarly to Gemma, including distillation methods that use intermediate data representations or methods based on the generation of synthetic data Outputs by Gemma" — subject to Gemma's prohibited-use policy. Gemma 4: Apache-2.0, no competitive-training clause. | Google | Gemma 3: Prohibited-Use policy attaches to all derivatives. Gemma 4: None. | **Gemma 3 is unusable. Gemma 4 is usable but still requires attribution; product brand-collision risk.** |
| **GPT-4 class (OpenAI)** | OpenAI ToS | **No** — "You may not use Output to develop models that compete with OpenAI" | OpenAI | Strict | **Unusable for our base/SFT pipeline.** |
| **Claude (Anthropic)** | Anthropic Usage Policy | **No** — "use the Services to develop products that compete with Anthropic" | Anthropic | Strict | **Unusable for our base/SFT pipeline.** |
| **Mistral / Ministral 3** (Apache-2.0) [4] | Apache-2.0 | Yes | User | None | Clean alternative to DeepSeek if a smaller teacher is wanted |
| **Qwen 3** (Apache-2.0) [5] | Apache-2.0 | Yes | User | None | Clean alternative; multilingual strength is an asset for our Hindi slot |

**Recommendation:** designate **DeepSeek-V3.2** as the primary teacher (MIT, no clause), with **Qwen-3** as a backup for multilingual synthesis (Apache-2.0). Forbid Llama- or Gemma-3- derived outputs in `src/myllm/post_train/`. Allow Gemma 4 outputs only with explicit attribution in the model card.

### 4.3 Eval coverage gaps

The PLAN.md §9 list is solid for 2024-vintage evaluations and missing several 2025-vintage standards. Adding:

| Gap | Recommended eval | Phase |
|---|---|---|
| Long-context retrieval beyond NIAH | **RULER** [34] @ 4k/8k for base; @ 32k post-anneal | 12 |
| Long-context reasoning | **BABILong** [35] @ 8k/16k | 12 |
| Long-context generation | **LongGenBench** | 12 |
| Long-context lexical-similarity-resistant | **NoLiMa** [36] | 12 |
| Multilingual reasoning | **MMLU-ProX** [32] (29 langs incl. Hindi) | 12 |
| Multilingual reading comp | **Belebele** [33] (122 langs) | 12 |
| Multilingual knowledge | **Global-MMLU** + **Global-MMLU-Lite** [33] | 12 |
| Indic-specific knowledge | **MILU** (already listed) — **promote from supplementary to gate** given 12% multilingual budget | 12 |
| Code (agentic / repo-level) | **SWE-bench Verified Lite** or **SWE-bench-Multimodal**: at 1B we expect <5% but the number is informative | 12 |
| Code (function-level, contamination-resistant) | **LiveCodeBench v6+** [30] | 12 |
| Agentic tool-use | **BFCL v3** (Berkeley Function Calling Leaderboard) | 12 |
| Jailbreak robustness | **JailbreakBench** [37] (replaces AdvBench, more rigorous) | 12 |
| Over-refusal | **XSTest** | 12 |
| Reasoning hard | **GPQA Diamond** (we are below the threshold to score well but the number matters for transparency) [46] | 12 |
| BBH replacement / harder | **BIG-Bench Extra Hard (BBEH)** | 12 |
| Instruction following | already have IFEval; add **MultiIF** for multilingual IF | 12 |

We do not need to *pass* all of these; we need to *report* all of them.

### 4.4 Regulatory posture

**EU AI Act / GPAI [38]:**
- Our model: ~3 × 10²² FLOP at 1B / 1T tokens (vs 10²⁵ threshold for "systemic risk"). We are **3 orders of magnitude below systemic-risk**.
- We are nonetheless a **GPAI provider** the moment we open-weight release. Obligations: technical documentation, training-data summary, copyright compliance, info-sharing with downstream users.
- Enforcement powers begin 2 Aug 2026. We will likely ship within this window.
- Concrete action: `docs/eu_ai_act_disclosure.md` (R9 above) + a public training-data summary table.

**DPDP Act (India) [47]:**
- Notified rules 13 Nov 2025; compliance deadline 13 May 2027.
- We are a "Data Fiduciary" the moment we collect or process Indian personal data. Pretraining on public Indian web data is contested — DPDP does not exempt "publicly accessible" data the way some Western regimes do.
- Mitigation: PII redaction at filter time (already planned in `pretrain_mix.yaml`), do not store user PII during serving without consent, document the lawful basis (consent / legitimate use) for any Indian-user-facing serving.

**NIST AI RMF + GAI Profile [48]:**
- Voluntary in the US but cited as compliance benchmark by multiple regulators.
- Map our governance to the GOVERN-MAP-MEASURE-MANAGE structure. Specifically: the gates in §4 of PLAN.md align with MEASURE; the red-team in Phase 12 aligns with MANAGE.

**UK AISI / Singapore AISI / US AISI** frameworks: voluntary; their public evaluation suites (UK Inspect, Singapore AICATCH) are useful eval-suite supplements but no obligation attaches.

### 4.5 Reproducibility / supply-chain integrity

Enterprise releases in 2025 standardly ship three artefacts. We should commit to all three by Phase 13.

1. **Model card** (`MODEL_CARD.md`) — Architecture, training compute, evaluation table per §4.3 above, intended use, out-of-scope use, known biases, license. Adopt the OLMo 2 or Granite 4 template.
2. **Data card** (`DATA_CARD.md`) — Per-dataset: source, version pin, retrieval date, license, attribution string, filter pipeline applied, % retained, decontamination summary. Adopt the FineWeb 2 template.
3. **Training card** (`TRAINING_CARD.md`) — Hyperparameters per phase, schedule, mesh config, throughput, MFU, total FLOP, total CO₂e (use the ML CO₂ Impact calculator), spike/rollback log.

Supply chain:
- Pin every Python dep in `requirements.txt` to an exact version + hash (`pip-compile --generate-hashes`).
- Sign Orbax checkpoints with cosign or sigstore. Cite hash in the model card.
- Cryptographically sign released weights at Phase 13 — Granite 4 does this and it is becoming table stakes [13].
- Maintain `manifests/tokenizer_v2.sha256` (already in `configs/pilot_250m.yaml`) — extend the pattern to weights.

---

## Section 5 — References

[1] Meta AI. "Llama 3.2: Revolutionizing edge AI and vision with open, customizable models." 2024. https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/ ; Model card: https://huggingface.co/meta-llama/Llama-3.2-1B

[2] Gemma Team, Google DeepMind. "Gemma 3 Technical Report." arXiv:2503.19786. 2025. https://arxiv.org/abs/2503.19786

[3] Microsoft. "Phi-4-Mini Technical Report: Compact yet Powerful Multimodal Language Models via Mixture-of-LoRAs." arXiv:2503.01743. 2025. https://arxiv.org/abs/2503.01743 ; Model card: https://huggingface.co/microsoft/Phi-4-mini-instruct

[4] Mistral AI. "Introducing Mistral 3 — Ministral 3-3B-Instruct-2512." Dec 2025. https://mistral.ai/news/mistral-3 ; Model card: https://docs.mistral.ai/models/ministral-3-3b-25-12

[5] Qwen Team, Alibaba. "Qwen3 Technical Report." arXiv:2505.09388. 2025. https://arxiv.org/abs/2505.09388 ; Model card: https://huggingface.co/Qwen/Qwen3-0.6B

[6] Hu, S. et al. "MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies." arXiv:2404.06395. 2024. https://huggingface.co/openbmb/MiniCPM3-4B

[7] DeepSeek-AI. "DeepSeek-V3.2." Dec 2025. https://huggingface.co/deepseek-ai/DeepSeek-V3.2 ; License: https://github.com/deepseek-ai/DeepSeek-V3/blob/main/LICENSE-MODEL

[8] Hugging Face SmolLM Team. "SmolLM2: When Smol Goes Big." 2024. https://huggingface.co/blog/smollm2

[9] Hugging Face. "SmolLM3: smol, multilingual, long-context reasoner." Jul 2025. https://huggingface.co/blog/smollm3 ; Model card: https://huggingface.co/HuggingFaceTB/SmolLM3-3B

[10] Allen AI. "OLMo 2: The best fully open language model to date." 2024-2025. https://allenai.org/blog/olmo2 ; 1B variant: https://huggingface.co/allenai/OLMo-2-0425-1B

[11] Bellagente, M. et al. "Stable LM 2 1.6B Technical Report." arXiv:2402.17834. 2024. https://arxiv.org/abs/2402.17834

[12] IBM. "Granite 3.0: Open, State-of-the-Art Enterprise Models." Oct 2024. https://www.ibm.com/new/announcements/ibm-granite-3-0-open-state-of-the-art-enterprise-models

[13] IBM. "Granite 4.0: Hyper-efficient, High Performance Hybrid Models for Enterprise." Oct 2025. https://www.ibm.com/new/announcements/ibm-granite-4-0-hyper-efficient-high-performance-hybrid-models

[14] Sarvam AI. "Sarvam-1: The first Indian language LLM." 2024. https://www.sarvam.ai/blogs/sarvam-1 ; Model card: https://huggingface.co/sarvamai/sarvam-1

[15] Wang, L. et al. "Pangu Pro MoE: Mixture of Grouped Experts for Efficient Sparsity." 2025. Curriculum methodology summarised in arXiv:2510.06826 (Mid-Training Survey).

[16] Yang, B. et al. "Rope to Nope and Back Again: A New Hybrid Attention Strategy." arXiv:2501.18795. 2025. https://arxiv.org/abs/2501.18795

[17] Gloeckle, F. et al. "Better & Faster Large Language Models via Multi-token Prediction." ICML 2024 / extensions through 2025. ACL Anthology "Pre-Training Curriculum for Multi-Token Prediction in Language Models" 2025.acl-long.1243. https://aclanthology.org/2025.acl-long.1243/ ; Google Gemma MTP drafters: https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/

[18] Fishman, M. et al. "Scaling FP8 Training to Trillion-Token LLMs." ICLR 2025. https://proceedings.iclr.cc/paper_files/paper/2025/file/f48b5133e89854a9e97cc22a6db83f25-Paper-Conference.pdf

[19] Fishman, M. et al. "To FP8 and Back Again: Quantifying Reduced Precision Effects on LLM Training Stability." arXiv:2405.18710. 2024. https://arxiv.org/abs/2405.18710

[20] Tian, C., Wang, J. et al. "WSM: Decay-Free Learning Rate Schedule via Checkpoint Merging for LLM Pre-training." arXiv:2507.17634v2. Aug 2025. https://arxiv.org/abs/2507.17634

[21] Hugging Face FineWeb Team. "FineWeb2: One Pipeline to Scale Them All — Adapting Pre-Training Data Processing to Every Language." Paper page: https://huggingface.co/papers/2506.20920 ; Dataset: https://huggingface.co/datasets/HuggingFaceFW/fineweb-2

[22] Su, D. et al. "Nemotron-CC: Transforming Common Crawl into a Refined Long-Horizon Pretraining Dataset." arXiv:2412.02595. ACL 2025. https://arxiv.org/abs/2412.02595 ; NVIDIA blog: https://developer.nvidia.com/blog/announcing-nemotron-cc-a-trillion-token-english-language-dataset-for-llm-pretraining/

[23] Yang, G. et al. "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer." NeurIPS 2021 / arXiv:2203.03466. https://arxiv.org/abs/2203.03466 ; EleutherAI practitioner guide: https://blog.eleuther.ai/mutransfer/

[24] Wang, G. et al. "FlashMask: Efficient and Rich Mask Extension of FlashAttention." ICLR 2025. https://arxiv.org/abs/2410.01359

[25] PyTorch Team. "FlexAttention + FlashAttention-4: Fast and Flexible." PyTorch Blog 2025. https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/

[26] Wortsman, M. et al. "Methods of improving LLM training stability." arXiv:2410.16682. 2024. https://arxiv.org/abs/2410.16682 ; "HybridNorm: Towards Stable and Efficient Transformer Training via Hybrid Normalization." arXiv:2503.04598. 2025.

[27] AI4Bharat. "Sangraha — Largest high-quality cleaned Indic language pretraining dataset (251B tokens, 22 languages)." CC-BY-4.0. https://huggingface.co/datasets/ai4bharat/sangraha

[28] TII. "Falcon RefinedWeb." ODC-By 1.0. https://huggingface.co/datasets/tiiuae/falcon-refinedweb

[29] BigCode. "The Stack v2." https://huggingface.co/datasets/bigcode/the-stack-v2 ; Opt-out repo discussion: https://huggingface.co/datasets/bigcode/the-stack/discussions/9

[30] Jain, N. et al. "LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code." arXiv:2403.07974. https://livecodebench.github.io/

[31] JAX Documentation. "jax.nn.dot_product_attention." https://docs.jax.dev/en/latest/_autosummary/jax.nn.dot_product_attention.html

[32] Wang, W. et al. "MMLU-ProX: A Multilingual Benchmark for Advanced Large Language Model Evaluation." arXiv:2503.10497. 2025. https://arxiv.org/abs/2503.10497

[33] Singh, S. et al. "Global-MMLU: Improving Multilingual Evaluation of LLMs." 2024-2025. https://llmdb.com/benchmarks/global-mmlu-lite ; Belebele: Bandarkar, L. et al. "The Belebele Benchmark: a Parallel Reading Comprehension Dataset in 122 Language Variants."

[34] Hsieh, C. et al. "RULER: What's the Real Context Size of Your Long-Context Language Models?" NVIDIA. https://github.com/NVIDIA/RULER

[35] Kuratov, Y. et al. "BABILong: a long-context needle-in-a-haystack benchmark for LLMs." https://github.com/booydar/babilong

[36] Modarressi, A. et al. "NoLiMa: Long-Context Evaluation Beyond Literal Matching." arXiv:2502.05167. 2025. https://arxiv.org/abs/2502.05167

[37] Chao, P. et al. "JailbreakBench: An Open Robustness Benchmark for Jailbreaking Language Models." NeurIPS 2024. https://jailbreakbench.github.io/

[38] European Commission. "EU rules on general-purpose AI models start to apply." Aug 2025. https://digital-strategy.ec.europa.eu/en/news/eu-rules-general-purpose-ai-models-start-apply-bringing-more-transparency-safety-and-accountability ; AI Office GPAI guidelines overview: https://artificialintelligenceact.eu/gpai-guidelines-overview/

[39] Meta. "Llama 3.2 Community License Agreement." https://www.llama.com/llama3_2/license/

[40] Google. "Gemma Terms of Use." https://ai.google.dev/gemma/terms ; "Gemma Prohibited Use Policy." https://ai.google.dev/gemma/prohibited_use_policy ; Gemma 4 Apache-2.0 discussion: https://www.mindstudio.ai/blog/what-is-gemma-4-apache-2-license-commercial-ai-deployment

[41] Penedo, G. et al. "The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale." arXiv:2406.17557. 2024. https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu (ODC-By 1.0)

[42] Rae, J. et al. "Compressive Transformers for Long-Range Sequence Modelling." PG-19 Dataset. Apache-2.0. https://github.com/google-deepmind/pg19

[43] Allen AI. "peS2o: Pretraining Efficiently on S2ORC!" ODC-BY. https://huggingface.co/datasets/allenai/peS2o ; Dolma license switch: https://blog.allenai.org/making-a-switch-dolma-moves-to-odc-by-8f0e73852f44

[44] EleutherAI. "Proof-Pile-2." Component-licensed; per-source attribution. https://huggingface.co/datasets/EleutherAI/proof-pile-2

[45] Allen AI / Google. "mC4." ODC-By 1.0 + Common Crawl ToU. https://huggingface.co/datasets/allenai/c4

[46] Rein, D. et al. "GPQA: A Graduate-Level Google-Proof Q&A Benchmark." 2023. Diamond subset used as 2025-standard hard reasoning eval.

[47] Government of India. "Digital Personal Data Protection Rules, 2025." Notified 13 Nov 2025. Discussion: https://www.deloitte.com/in/en/services/consulting/about/indias-dpdp-rules-2025-leading-digital-privacy-compliance.html ; AI training data analysis: https://www.khuranaandkhurana.com/ai-training-data-under-india-s-dpdp-regime-compliance-challenges-and-strategies

[48] NIST. "AI Risk Management Framework + Generative AI Profile (NIST AI 600-1)." https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf ; 2025 updates: https://www.nist.gov/itl/ai-risk-management-framework

---

*End of dossier. Total: ~5,100 words of audited content. Author: research role; reviewer: project owner (harshit.hv@samatva.com). Next refresh trigger: any of (a) Phase 2 pilot completion + Gate 1 review, (b) public release of a 1-3B-class model with materially new architecture (e.g., a fully-Mamba 1B), (c) substantive change to the EU AI Act enforcement regime or DPDP rules, (d) 2026-09 — 4 months out — as a calendar checkpoint.*
