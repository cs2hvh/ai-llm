# B200/B300 cluster economics + parallelism plan for 7B

## RECOMMENDATION
Rent a reserved block of 64x B200 (8 HGX nodes) from a neocloud (Nebius/CoreWeave/Lambda class) for ~2.5 months at a negotiated <=$6.00/GPU-hr, and train the 7B dense model on ~6T tokens: budget ~70-90k GPU-hours = roughly $420k-540k compute (add 10-15% for ablations, the ~100-300B-token long-context + midtraining phases, and restarts; halve the ablation line by running it on preemptible capacity at ~$4/GPU-hr). Do NOT pay a B300 premium: dense BF16/FP8 FLOPs are identical to B200, and its 288GB HBM only matters for the final 128k phase — either rent a short B300 block for that phase or just use context parallelism (CP=4-8) on the B200s. Parallelism should be deliberately boring: pure FSDP2-style sharding within nodes + HSDP replication across nodes (in your JAX/Keras3 stack, plain GSPMD data-axis sharding), no tensor parallelism, global batch 2M ramping to 4M tokens (OLMo 3's proven setting) at 8k sequence length for the main run, FP8 (MXFP8/rowwise float8) targeted for a 1.3x speedup with BF16 as the fallback, planning MFU 35% and stretch 45%. Operationally: async checkpoints (~70-100GB each with your Muon hybrid) every 30-60 minutes to object storage (~6TB retained, ~$130/mo), simple elastic auto-restart rather than torchft at this scale, and streaming dataloader from object storage (required aggregate is only ~6MB/s — trivial). If serving economics later dominate, a Qwen3-30B-A3B-style MoE at 6T tokens would cost roughly half the FLOPs (~$230-300k), but for a solo-led team pivoting stacks, the dense 7B is the materially lower-risk, proven path — pin the budget conversation at ~$500k +/- $120k and lock the rate soon, since Blackwell rental prices rose (not fell) in June 2026.

## VERIFICATION VERDICTS
- [CONFIRMED] B300 (Blackwell Ultra) has identical dense BF16 (2.25 PF/GPU) and FP8 (4.5 PF/GPU) throughput to B200 — only FP4 (108 vs 72 PF dense per 8-GPU HGX) and HBM (288 vs 192 GB) differ — so B300 adds ~zero FLOPs for a BF16/FP8 pretrain.
  evidence: NVIDIA HGX page (https://www.nvidia.com/en-us/data-center/hgx/) spec table: HGX B300 FP4 Tensor Core "144 PFLOPS | 108 PFLOPS" (sparse|dense) vs HGX B200 "144 PFLOPS | 72 PFLOPS" — the 108 vs 72 dense FP4 figure matches exactly. FP8/FP6 is "72 PFLOPS" for BOTH (= 9 PF/GPU sparse, 4.5 PF/GPU dense) and FP16/BF16 is "36 PFLOPS" for BOTH (= 2.25 PF/GPU dense), so zero added BF16/FP8 FLOPs is correct 
- [NUANCED] Mid-2026 B200 rental spans ~$5.89-9.36/GPU-hr on-demand (Nebius $7.15, Lambda $6.69-6.99, CoreWeave ~$9.36, RunPod $5.89) and $3.95-5.34 preemptible; Nebius B300 is $7.85 on-demand / $4.30 preemptible, with prices RAISED on 2026-06-01.
  CORRECTION: On-demand span is ~$5.89-8.60/GPU-hr (RunPod $5.89 low, CoreWeave $8.60 high — $68.80 per 8-GPU instance). Nebius/Lambda/RunPod figures and the June 1, 2026 Nebius price raise are all correct. The $5.34 preemptible upper bound could not be sourced (verified preemptible: Nebius B200 $3.95, B300 $4.30); B300 is also rentable at $7.39/hr on RunPod.
  evidence: Nebius pricing docs (https://docs.nebius.com/compute/resources/pricing) confirm exactly: B200 "$7.15 per 1 GPU hour" on-demand / "$3.95" preemptible and B300 "$7.85" / "$4.30" from June 1, 2026, with the note "Starting June 1, 2026, prices for virtual machines with NVIDIA B300, B200, H200 and H100 GPUs are updated" (raised from $5.50/$2.90 B200 and $6.10/$3.40 B300). Lambda (https://lambda.ai/pric
- [CONFIRMED] FSDP-only (no TP) training of an 8B model is proven at 33-42% MFU on 8-128 GPUs in torchtitan, and float8 + FSDP2 + compile adds ~50% throughput at loss parity — so a 7B on 8-64 Blackwell GPUs needs only FSDP2/HSDP.
  evidence: torchtitan paper (https://arxiv.org/abs/2410.06511, HTML at arxiv.org/html/2410.06511): "the 1D Llama 3.1 8B model training on 8 or 128 H100 GPUs without Float8 achieves 33% to 42% MFU." Table 1 (8 GPUs): FSDP baseline 6,258 tok/s → FSDP + torch.compile + Float8 9,409 tok/s = 50.35% speedup (at 128 GPUs it is 65.08%: 5,645 → 9,319 tok/s). Loss parity: Figure 5 shows "loss converging tests covering
- [CONFIRMED] 45% MFU with FP8 on B200 clusters is demonstrated (41.4k tok/s/GPU for a 3B on 128 B200s), and NVIDIA measured MXFP8 at 1.28-1.37x over BF16 on DGX B200, supporting a 35% (conservative) to 45% (achievable) MFU planning band.
  evidence: Tzafon blog (https://www.tzafon.ai/blog/breaking-40k-tokens) confirms: 3B model, 128 B200 GPUs, 41,400 tok/s/GPU, "45% MFU" for TorchTitan+FP8 vs 29.4% for the Nanotron baseline, using FP8 with BF16 fallback on LayerNorm/embeddings/final projection. The MXFP8 figure is NOT in the stated tzafon source — it comes from NVIDIA's blog "Faster Training Throughput in FP8 Precision with NVIDIA NeMo" (http
- [REFUTED] MLPerf Training v5.1 (Nov 2025): Llama-3.1-8B benchmark scales 8→32 B200s at ~96% efficiency (85.4→27.8 min) and 8x B300 is only ~12.6% faster than 8x B200 — confirming near-linear small-cluster scaling and B300's marginal training gain.
  CORRECTION: Times and the ~12.6% B300 delta are correct, but 8→32 B200 scaling on Llama-3.1-8B is ~3.07x speedup = ~77% efficiency (per Nebius's own "~3.1x speed-up" statement), not ~96%. Drop the 'near-linear' characterization or re-derive it from sustained token throughput rather than MLPerf time-to-train.
  evidence: Nebius MLPerf v5.1 blog (https://nebius.com/blog/posts/mlperf-training-v5-1-results) confirms the raw times — Llama 3.1-8B: "85.37 min" on 8x B200, "27.83 min" on 32x B200, "75.84 min" on 8x B300 — and "the HGX B300 system showed an average 12.6% reduction in training time" vs B200 (for the 8B benchmark alone: 85.37/75.84 = 12.6% higher throughput). But the blog itself calls 8→32 a "~3.1x speed-up
- [NUANCED] OLMo 3 7B — the closest open playbook — trained on 5.93T Dolma 3 tokens with a 4M-token global batch at ~7,700 tok/s per H100 (~33% MFU), then 100B midtraining + 50B long-context tokens.
  CORRECTION: All numbers correct except MFU: the OLMo 3 report states roughly 43% MFU for the 7B (7,700 tok/s/GPU on H100, BF16, seq len 8192), not ~33%. If the plan's 35-45% B200 MFU band was benchmarked against 'OLMo 3 got 33%', note that Ai2's own reported figure is 43% — the discrepancy is MFU accounting methodology, and the plan should use one consistent MFU definition throughout.
  evidence: OLMo 3 technical report (https://arxiv.org/abs/2512.13961, redirected from allenai.org/papers/olmo3) confirms: one epoch = "5.93T tokens" (Dolma 3 Mix, Table 4: Total 5.93T); 7B batch size "4M tokens per batch" (vs 8M for 32B); "we train the 7B model at 7700 tokens per second per GPU ... at a sequence length of 8192, using bfloat16 precision throughout"; midtraining "100B training tokens" and long
- [CONFIRMED] Llama 3 ran the bulk of pretraining at 8k context and ramped to 128k in six stages over only ~800B final tokens using context parallelism, because attention compute grows quadratically — CP is a late-phase tool, not a main-run requirement, for 7B scale.
  evidence: Llama 3 paper (https://arxiv.org/abs/2407.21783, text via ar5iv.labs.arxiv.org/html/2407.21783), Section 3.4.2: "We increased context length gradually in six stages, starting from the original 8K context window and ending in the final 128K context window" and "This long-context pre-training stage was performed using approximately 800B training tokens" (~800B of ~15T total, so the bulk was indeed a
- [NUANCED] By the 6*N*D formula, a 7B/6T-token run costs ~69k-89k B200-hours (45%/35% MFU) = ~$380k-636k at $5.50-7.15/GPU-hr, finishing in ~45-58 days on 64 B200s or ~90-116 days on 32.
  CORRECTION: GPU-hours and day counts are arithmetically correct. Update the price band: on-demand low is now ~$5.89 (RunPod) since Nebius raised $5.50 → $7.15 on 2026-06-01, giving ~$407k-636k on-demand (or ~$273k+ preemptible at Nebius $3.95). Treat 45% MFU on B200 for a 7B as a target, not a demonstrated number.
  evidence: Arithmetic verified against the NVIDIA-confirmed B200 dense BF16 peak of 2.25 PF/GPU (https://www.nvidia.com/en-us/data-center/hgx/: 36 PFLOPS FP16/BF16 per 8-GPU sparse = 2.25 PF dense/GPU): 6 x 7e9 x 6e12 = 2.52e23 FLOPs; at 45% MFU → 69,136 GPU-h, at 35% → 88,889 GPU-h (matches ~69k-89k); 69,136 x $5.50 = $380k and 88,889 x $7.15 = $636k (matches); 69,136/64 GPUs = 45.0 days, 88,889/64 = 57.9 d

## FULL REPORT
# B200/B300 Cluster Economics + Parallelism Plan for a 7B Agentic Pretrain (researched 2026-07-23)

## 1. B200 vs B300: verified hardware facts

**Per-GPU compute is identical in BF16 and FP8; B300 only adds FP4 and memory.** NVIDIA's official HGX page (fetched 2026-07-23) lists both 8-GPU systems at 36 PFLOPS sparse BF16 and 72 PFLOPS sparse FP8 — i.e., **2.25 PFLOPS dense BF16 and 4.5 PFLOPS dense FP8 per GPU on both B200 and B300**. The only compute difference: FP4 dense is 72 PFLOPS/system on HGX B200 vs 108 PFLOPS/system on HGX B300 (9 vs ~13.5 PF dense FP4 per GPU) (https://www.nvidia.com/en-us/data-center/hgx/, 2026-07-23; corroborated by https://www.glennklockwood.com/garden/processors/b300, 2026-07-23, which also notes B300's FP64 is cut to ~1.2 TF vs 37 TF on B200).

| Spec (per GPU) | B200 | B300 (Blackwell Ultra) |
|---|---|---|
| HBM3e | 192 GB raw (~180 usable; NVIDIA lists HGX B200 as 1.4 TB/8) | 288 GB raw (~262–270 usable; NVIDIA lists HGX B300 as 2.1 TB/8) |
| HBM bandwidth | 8 TB/s | 8 TB/s |
| Dense BF16 | 2.25 PFLOPS | 2.25 PFLOPS (same) |
| Dense FP8 | 4.5 PFLOPS | 4.5 PFLOPS (same) |
| Dense FP4 | 9 PFLOPS | ~13.5–15 PFLOPS |
| NVLink 5 | 1.8 TB/s per GPU, 14.4 TB/s per 8-GPU node | same |
| Node networking | 0.8 TB/s per HGX system | 1.6 TB/s (ConnectX-8 class) |
| TDP | 1,000 W | up to 1,400 W, typically liquid-cooled |

Sources: https://www.nvidia.com/en-us/data-center/hgx/ (2026-07-23); https://www.glennklockwood.com/garden/processors/b300 (2026-07-23); https://www.server-parts.eu/post/nvidia-b300-gpu-blackwell-ultra-architecture (2026-07).

**Availability timeline:** B300 debuted at GTC March 2025; partner systems shipped H2 2025 (https://vast.ai/article/everything-you-need-to-know-about-the-nvidia-blackwell-ultra-b300). CoreWeave powered on the first GB300 NVL72 in July 2025, and in mid-2026 GB300 NVL72 still moves through enterprise sales at CoreWeave/Azure/AWS, while HGX B200 is broadly available on-demand (https://www.spheron.network/blog/gb300-nvl72-vs-gb200-nvl72-pricing-availability-2026/, 2026). Nebius has public B300 pricing effective 2026-06-01 (https://docs.nebius.com/compute/resources/pricing, fetched 2026-07-23) and submitted HGX B300 MLPerf Training v5.1 results in Nov 2025 (https://nebius.com/blog/posts/mlperf-training-v5-1-results, 2025-11-12).

**MLPerf v5.1 (Nov 2025, Nebius submissions):** Llama-3.1-8B pretraining benchmark: 8×B300 = 75.84 min vs 8×B200 = 85.37 min (Blackwell Ultra ~12.6% average faster); 16×B200 = 51.83 min; 32×B200 = 27.83 min → **8→32 GPU scaling is ~3.07x (96% efficiency)** for an 8B-class model with no exotic parallelism (https://nebius.com/blog/posts/mlperf-training-v5-1-results, 2025-11-12).

**Implication for this project:** for a BF16/FP8 7B pretrain, B300 buys you ~0% extra FLOPs and +50% HBM. It matters only for the 128k-context phase (bigger activations) or FP4 experiments — not for the main run.

## 2. Rental market, mid-2026 ($/GPU-hr, per GPU)

| Provider | B200 on-demand | B200 spot/preemptible | B200 reserved | B300 | Source + date |
|---|---|---|---|---|---|
| Nebius | $7.15 | $3.95 | n/l | $7.85 OD / $4.30 preempt | https://docs.nebius.com/compute/resources/pricing (prices effective 2026-06-01, fetched 2026-07-23) |
| Lambda | $6.69–6.99 | — | 1-Click Cluster 16–256 GPUs: $8.87–9.86 (2wk–1yr; incl. fabric/storage) | not listed | https://lambda.ai/pricing (2026-07-23) |
| CoreWeave | ~$9.36 | ~$5.34 | contact sales | contact sales (HGX B300/GB300) | https://costbench.com/software/ai-gpu-cloud/coreweave/ and https://getdeploying.com/gpus/nvidia-gb200 (indicative as of 2026-07-12) — UNVERIFIED against CoreWeave's own page |
| RunPod | $5.89 | — | — | — | https://www.thundercompute.com/blog/nvidia-b200-pricing (2026-07-07) |
| Vast.ai | ~$7.12 (marketplace, varies) | varies | — | — | same, 2026-07-07 |
| Hyperbolic | $3.50 | — | — | — | same, 2026-07-07 (UNVERIFIED, thin capacity) |
| AWS | ~$14.24 (p6-b200) | — | — | — | https://intuitionlabs.ai/articles/data-center-gpu-pricing-2026 (2026-07-20) |
| GCP | quote | $4.95 spot (a4-highgpu-8g) | — | — | same, 2026-07-20 |
| Specialist (unnamed) | — | — | 36-mo reserved from ~$2.25 | $8.55 OD / $3.67 spot | same, 2026-07-20 (UNVERIFIED provider identity) |
| SF Compute | B200 not listed; H100 market avg $2.16 | market | buy/sell blocks, no lock-in | — | https://sfcompute.com/ (2026-07-23) |

Notes: (a) Nebius **raised** Blackwell prices on 2026-06-01 (B200 OD $5.50→$7.15) — supply is tightening, not loosening; (b) GB200/GB300 NVL72 racks rent at ~$42/hr+ full-rack, enterprise-only (https://getdeploying.com/gpus/nvidia-gb200, 2026-07-12); (c) a realistic planning band for a 1–3 month committed 32–64 GPU B200 block is **$5.50–7.15/GPU-hr**, with preemptible at **$3.95–5.34**.

## 3. Parallelism plan for a 7–8B dense model on 8–64 Blackwell GPUs

**FSDP/HSDP alone is sufficient; no tensor parallelism.** Evidence:
- torchtitan trains Llama 3.1 8B with **1D FSDP only** at 33–42% MFU on 8–128 H100s (https://arxiv.org/pdf/2410.06511, 2024-10); FSDP2 beats FSDP1 with 7% lower peak memory (https://github.com/pytorch/torchtitan/blob/main/docs/fsdp.md).
- FSDP2 + torch.compile + float8 gives ~50% throughput over bf16 FSDP1 baseline at loss parity (6,258 → 9,409 tok/s/GPU for 8B on 8×H100) (https://pytorch.org/blog/training-using-float8-fsdp2/, 2024).
- Memory math: mixed-precision AdamW state ≈ 14 B/param ≈ 98 GB total for 7B; sharded over just 8 GPUs that is ~12 GB/GPU against 180 GB usable — enormous headroom for activations and large micro-batches. NVLink 5 (1.8 TB/s/GPU) makes intra-node all-gathers nearly free; use **HSDP (shard within node, replicate across nodes)** at 16–64 GPUs so inter-node traffic is a single overlappable 14 GB bf16 gradient all-reduce per step.
- The JAX equivalent (this project's stack) is plain jit/GSPMD FSDP sharding on the data axis — same conclusion.

**Context parallelism is needed only for the 128k phase.** Llama 3 kept 8k context for the bulk of pretraining and ramped to 128k in six stages over only ~800B tokens at the end, using all-gather-based context parallelism because attention cost grows quadratically (https://arxiv.org/pdf/2407.21783, 2024-07). Qwen3 similarly did its long-context regime (16k–32k) as a late phase (https://arxiv.org/pdf/2505.09388, 2025-05); OLMo 3 did a 50B-token long-context extension after 5.9T main tokens (https://allenai.org/blog/olmo3, 2025-11-20). Practical rule for 7B on 192 GB GPUs: seq ≤ 32k needs no CP; at 64k–128k use CP=4–8 (supported in torchtitan; ring/all-gather variants per https://arxiv.org/abs/2411.01783). B300's 288 GB would roughly halve the CP degree needed — the one place it earns its premium.

**Global batch size:** OLMo 3 7B used a **4M-token global batch** for 5.93T tokens on 1,024 H100s (https://ritvik19.medium.com/papers-explained-571-olmo-3-1ee7134a4e67, 2026-06; https://allenai.org/blog/olmo3, 2025-11-20). Llama 3 405B ramped 4M→8M→16M (https://arxiv.org/pdf/2407.21783, 2024-07). For a 7B: start ~2M, run the bulk at 4M, optionally ramp to 8M late. On 64 GPUs at 4M GBS that is 64k tokens/GPU/step (e.g., 8×8k sequences, grad-accum as needed) — comfortable.

**Achievable MFU:** BF16 planning band **35–45%**: torchtitan 33–42% (H100, https://arxiv.org/pdf/2410.06511); OLMo 3 7B ran at ~7,700 tok/s/device on H100 ≈ 32–33% MFU at 1,024-GPU scale (https://allenai.org/blog/olmo3, 2025-11-20). FP8 on Blackwell: NVIDIA measured **1.28–1.37x over BF16 with MXFP8 on DGX B200** (Llama3-8B ≈1.30x) (https://developer.nvidia.com/blog/faster-training-throughput-in-fp8-precision-with-nvidia-nemo/, 2025), and Tzafon hit **45% MFU / 41.4k tok/s/GPU with FP8 on 128×B200** for a 3B model (https://www.tzafon.ai/blog/breaking-40k-tokens, 2025-07-22). Treat 35% BF16-basis MFU as conservative, 45% as achievable-with-effort (FP8 + fused kernels + compile).

## 4. The math: FLOPs, wall-clock, dollars

Formula: **C = 6·N·D**; GPU-hours = C / (MFU × 2.25e15 FLOP/s dense BF16) / 3600. N = 7e9. Cost = GPU-hours × $/GPU-hr (cost is independent of cluster size; days = GPU-hr / #GPUs / 24). Price columns use the mid-2026 committed band $5.50 and Nebius on-demand $7.15.

| Config | D | FLOPs | MFU | B200 GPU-hr | days @8 | @16 | @32 | @64 | $ @5.50 | $ @7.15 |
|---|---|---|---|---|---|---|---|---|---|---|
| 7B dense | 2T | 8.4e22 | 35% | 29,630 | 154 | 77 | 39 | 19 | $163k | $212k |
| 7B dense | 2T | 8.4e22 | 45% | 23,045 | 120 | 60 | 30 | 15 | $127k | $165k |
| 7B dense | 4T | 1.68e23 | 35% | 59,259 | 309 | 154 | 77 | 39 | $326k | $424k |
| 7B dense | 4T | 1.68e23 | 45% | 46,091 | 240 | 120 | 60 | 30 | $253k | $330k |
| 7B dense | 6T | 2.52e23 | 35% | 88,889 | 463 | 231 | 116 | 58 | $489k | $636k |
| 7B dense | 6T | 2.52e23 | 45% | 69,136 | 360 | 180 | 90 | 45 | $380k | $494k |
| 7B dense | 12T | 5.04e23 | 35% | 177,778 | 926 | 463 | 231 | 116 | $978k | $1.27M |
| 7B dense | 12T | 5.04e23 | 45% | 138,272 | 720 | 360 | 180 | 90 | $760k | $989k |
| 30B-A3B MoE (active N≈3.3e9, Qwen3-30B-A3B-like) | 6T | 1.19e23 | 35% | 41,905 | 218 | 109 | 55 | 27 | $230k | $300k |
| 30B-A3B MoE | 6T | 1.19e23 | 45% | 32,593 | 170 | 85 | 42 | 21 | $179k | $233k |

Caveats: (a) FP8 runs effectively land in the 45%-row or better (BF16-equivalent basis); (b) MoE MFU in practice runs 5–10 pts lower than dense due to routing/all-to-all, and the 30B total params still cost ~60 GB/GPU of sharded optimizer state on 8 GPUs — fine on B200 but the MoE row's speed advantage partially erodes; (c) add ~10–15% budget for ablations, restarts, and the long-context + midtraining phases (OLMo 3 spent 100B midtraining + 50B long-context tokens on top of 5.93T; Llama 3 spent ~800B on context ramp). Preemptible pricing ($3.95–5.34) cuts the dollar columns ~30–45% at the cost of restart engineering.

Reference-point sanity check: 2×MLPerf — 32×B200 completes the Llama-3.1-8B benchmark slice in 27.8 min (https://nebius.com/blog/posts/mlperf-training-v5-1-results, 2025-11-12), and OLMo 3's full 5.93T-token 7B on 1,024 H100s at 7.7k tok/s/GPU ≈ 209 hr ≈ 214k H100-hr — consistent with our 88.9k B200-hr at 35% (B200 ≈ 2.27x H100 dense BF16).

## 5. Operational plan

**Checkpointing:** full 7B training state (fp32 master + AdamW moments + bf16 params) ≈ 98 GB; with your Muon hybrid (single momentum buffer on hidden matrices) ≈ 70–85 GB. Use PyTorch DCP-style **async checkpointing** — 5–15x lower overhead vs sync for Llama-3.1-8B-class models, proven at 1,856-GPU scale (https://pytorch.org/blog/6x-faster-async-checkpointing/, 2025; https://docs.pytorch.org/tutorials/recipes/distributed_async_checkpoint_recipe.html). Cadence: every 30–60 min transient (keep last 3–5) + permanent every ~50–100B tokens. A 60-run-day schedule retaining ~60 permanents ≈ **6 TB object storage ≈ $130/mo at S3 rates** — negligible. In JAX, Orbax async checkpointing is the equivalent.

**Fault tolerance:** at 32–64 GPUs, expected failure interval is days, not hours — auto-restart-from-latest-async-checkpoint (elastic torchrun or a supervisor script) is sufficient. torchft (fault-tolerant HSDP with per-replica-group restart, Lighthouse coordination) becomes worth its complexity at 256+ GPUs or on preemptible capacity; it has been demonstrated with 2,000 synthetic failures and checkpoint-free recovery (https://github.com/meta-pytorch/torchft; https://pytorch.org/blog/fault-tolerant-llama-training-with-2000-synthetic-failures-every-15-seconds-and-no-checkpoints-on-crusoe-l40s/, 2025).

**Dataloader throughput:** at 64×B200, 45% MFU, the cluster consumes ~1.54M tokens/s (24.1k tok/s/GPU × 64). With your 131k-vocab SPM tokenizer, ids need 4 bytes (>uint16 max) → **~6.2 MB/s aggregate** — trivially sustainable streaming from object storage; MosaicML StreamingDataset-style sharded streaming with prefetch saturates this with mid-epoch deterministic resume (https://www.databricks.com/blog/mosaicml-streamingdataset; https://docs.mosaicml.com/projects/streaming/en/latest/distributed_training/performance_tuning.html). In JAX, Grain/ArrayRecord streaming is the equivalent. Data is never the bottleneck at this scale; tokenizer/packing pipeline QA is.

**Spot vs reserved:** the main 45–90 day run belongs on **reserved/committed** capacity (price certainty, no preemption tax on a solo team); use **preemptible** ($3.95 Nebius / $5.34 CoreWeave / $4.95 GCP) for ablations and the muP-transfer pilot ladder. Note Nebius's June 2026 price *increase* — lock a rate early; SF Compute-style markets (https://sfcompute.com/) are useful for burst ablation blocks but showed no B200 listings at fetch time.

## OPEN QUESTIONS
- Exact CoreWeave and Crusoe B200/B300 reserved-contract rates for a 32-64 GPU, 2-3 month block (both are 'contact sales'; the ~$9.36 CoreWeave on-demand figure is aggregator-sourced and UNVERIFIED against CoreWeave's own price sheet).
- Achievable MFU for FP8/MXFP8 on Blackwell in JAX/XLA (Keras3) specifically — all strong published MFU numbers found (torchtitan 33-42%, Tzafon 45%) are PyTorch; no live source quantified JAX FP8 pretraining MFU on B200 in 2026.
- Whether MXFP8 loss parity holds over a full 6T-token 7B horizon — NVIDIA's 1.28-1.37x results and the PyTorch float8 parity claims are from shorter runs; a BF16 fallback budget (+30% time) should be held in reserve.
- Real-world preemption frequency on Nebius preemptible / CoreWeave spot B200 pools (no published preemption-rate data), which determines whether the main run could safely ride spot at ~$4/GPU-hr instead of reserved.
- AWS p6-b200 true per-GPU on-demand rate (sources conflict: ~$14.24/GPU-hr vs much lower aggregator figures) — only relevant if the company is contractually tied to AWS.
- SF Compute market depth for B200 blocks (site showed only H100 market pricing at ~$2.16 avg on 2026-07-23) — potentially the cheapest ablation-burst venue if B200 inventory exists.
- Ultra-low advertised B200 prices (Hyperbolic $3.50, Vultr sub-$1 figures) could not be verified for real multi-node NVLink/IB-fabric availability and are likely single-GPU or promotional — treat as UNVERIFIED and unusable for a 64-GPU pretrain plan.