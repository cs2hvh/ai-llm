# Pre-2 Final Architecture and Training Plan

Date: 2026-05-14
Status: final planning decision for pre-2 implementation
Scope: 1B-class foundation model, at least 1T tokens, with a practical path to 1.5T-3T tokens

## 1. Final Decision

Pre-2 should be a new mainline training stack and recipe, not an incremental patch on the pre-1 Keras/JAX implementation.

Final mainline decision:

- Model: dense decoder-only Transformer, approximately 1.49B total parameters.
- Training target: 1.5T tokens for the pre-2 base release.
- Minimum viable run: 1.0T tokens, internal only unless evals are unexpectedly strong.
- Stretch run: 3.0T tokens if loss and eval curves continue to improve.
- Stack: PyTorch 2.12+ with TorchTitan, FSDP2, DTensor, and Distributed Checkpointing.
- Data: offline packed corpus remains a core asset, but the training loader must be rewritten as a TorchTitan/PyTorch dataloader with exact resumability.
- Distillation: disable the current cross-tokenizer top-K logit distillation path. Use teacher-generated text and reasoning traces unless teacher and student tokenizers are byte-for-byte identical.

MoE decision:

- Do not make MoE the pre-2 default.
- A 1B-total MoE is not worth the routing, load-balancing, and expert-parallel complexity.
- If MoE is pursued, it should be a separate research branch: approximately 1B active parameters and 6B-8B total resident parameters, OLMoE-style.

Pre-1 remains useful as:

- a pilot/canary stack;
- a correctness reference;
- a data and checkpointing lessons source;
- not the production pre-2 training path.

## 2. Why This Decision

The 2026 evidence is clear:

- The best small models are heavily overtrained relative to Chinchilla-style compute-optimal ratios.
- Dense 1B-2B models remain competitive when trained on high-quality trillion-token mixtures.
- MoE is valuable when total resident parameters are much larger than active parameters, not when total parameters are constrained near 1B.
- Mature distributed stacks now provide primitives this repo is currently hand-building.

Examples from current public systems:

- Llama 3.2 1B reports 1.23B parameters, GQA, shared embeddings, 128k context, and up to 9T pretraining tokens.
- SmolLM3 uses a dense decoder, GQA, staged data mixture, WSD, and 11.2T tokens at 3B scale.
- Qwen3 has strong dense small models and MoE models at larger active/total scales.
- OLMoE demonstrates the useful MoE regime: 1B active / 7B total, not 1B total.
- DeepSeek-V3 validates MoE, MLA, and multi-token prediction at frontier scale, but that does not imply those should be first-line complexity for a small-team 1.5B dense run.

## 3. Mainline Model: MyLLM Pre-2 Dense 1.5B

Target name:

```text
myllm-pre2-dense-1.5b-base
```

Architecture:

| Field | Decision |
|---|---:|
| Architecture | Decoder-only Transformer |
| Total parameters | approximately 1.49B |
| Non-embedding parameters | approximately 1.22B |
| Layers | 20 |
| Hidden size | 2048 |
| FFN size | 8192 |
| FFN activation | SwiGLU |
| Attention | GQA |
| Query heads | 32 |
| KV heads | 8 |
| Head dim | 64 |
| Norm | RMSNorm, pre-norm |
| Attention stabilization | QK-norm |
| Positional encoding | RoPE |
| RoPE base | 1,000,000 initial value, ablate 500,000 and 5,000,000 |
| Vocab | 131,075 runtime-token SentencePiece-Unigram artifact, subject to tokenizer ablation |
| Embeddings | tied input/output embeddings |
| Training dtype | BF16 |
| Baseline context | 8192 during foundation training |
| Final context target | 32768 after long-context continuation |

Important context decision:

Do not train the full 1.5T-token foundation phase at 32k context. That is expensive and not necessary. Train most tokens at 8k packed sequences, then run a dedicated long-context continuation stage at 16k and 32k.

## 4. Parameter Math

Definitions:

```text
L       = 20 layers
d       = 2048 hidden size
f       = 8192 FFN size
V       = 131075 runtime vocab size
Hq      = 32 query heads
Hkv     = 8 key/value heads
dh      = 64 head dim
d_kv    = Hkv * dh = 512
```

Embedding parameters, tied:

```text
V * d = 131075 * 2048 = 268,441,600
```

Attention parameters per layer:

```text
Q = d * d       = 2048 * 2048 = 4,194,304
K = d * d_kv    = 2048 * 512  = 1,048,576
V = d * d_kv    = 2048 * 512  = 1,048,576
O = d * d       = 2048 * 2048 = 4,194,304

attention_per_layer = 10,485,760
```

SwiGLU FFN parameters per layer:

```text
gate = d * f
up   = d * f
down = f * d

ffn_per_layer = 3 * d * f
              = 3 * 2048 * 8192
              = 50,331,648
```

Core transformer parameters:

```text
per_layer = attention_per_layer + ffn_per_layer + small_norm_terms
          ~= 60,821,504

20 layers ~= 1,216,430,080
```

Total:

```text
core + tied_embedding ~= 1,216,430,080 + 268,441,600
                     ~= 1,484,865,536

including norms and small terms: approximately 1.49B
```

If embeddings are untied, add another 268M parameters. Therefore embeddings must remain tied.

## 5. Training Compute Math

First-order dense Transformer training FLOPs:

```text
F_train ~= 6 * N * T
```

Where:

```text
N = 1.49e9 parameters
T = training tokens
```

Token budgets:

| Token budget | Tokens per parameter | Approx training FLOPs |
|---:|---:|---:|
| 1.0T | 671 | 8.9e21 |
| 1.5T | 1007 | 1.34e22 |
| 3.0T | 2013 | 2.68e22 |
| 5.0T | 3356 | 4.47e22 |

This formula is for planning. Long-context attention adds overhead, especially at 16k and 32k sequence lengths, which is why the long-context phase should be short and explicit.

Recommended global batch:

```text
global_batch_tokens ~= 2,097,152 tokens
```

Example with 8 GPUs at sequence length 8192:

```text
tokens_per_microstep = 8 GPUs * 1 sequence/GPU * 8192
                     = 65,536 tokens

grad_accum_steps = 32

global_batch_tokens = 65,536 * 32
                    = 2,097,152
```

Approximate optimizer steps:

| Token budget | Steps at 2.097M tokens/step |
|---:|---:|
| 1.0T | 477k |
| 1.5T | 715k |
| 3.0T | 1.43M |
| 5.0T | 2.38M |

## 6. Training Objective and Optimizer

Main objective:

```text
loss = CE(next_token) + z_loss
```

Baseline optimizer:

| Field | Decision |
|---|---:|
| Optimizer | AdamW |
| beta1 | 0.9 |
| beta2 | 0.95 |
| weight decay | 0.1 |
| gradient clipping | 1.0 |
| peak LR | 2e-4 initial baseline |
| LR ablations | 1.5e-4, 2e-4, 3e-4 |
| scheduler | WSD |
| warmup | 2k-5k steps |
| stable phase | 90 percent of main training |
| decay | final 10 percent, linear to 0 |
| z-loss | 1e-4 baseline, ablate 1e-5 and 1e-3 |

Muon decision:

- Do not make Muon the first production optimizer.
- Run Muon or MuonClip only as a 100B-token ablation after the AdamW baseline is stable.
- If Muon wins clearly on validation loss per GPU-hour without instability, promote it to pre-2.1.

Multi-token prediction decision:

- Do not include MTP in the first baseline run.
- Add it as an auxiliary-head ablation after the dense CE baseline reaches stable 50B-100B proxy training.
- If used, keep auxiliary heads removable so inference can fall back to standard next-token decoding.

## 7. Token Budget and Data Stages

Minimum acceptable internal run:

```text
1.0T tokens
```

Recommended pre-2 base release:

```text
1.5T tokens
```

Stretch:

```text
3.0T tokens
```

Research ceiling:

```text
5.0T tokens
```

Staged recipe for 1.5T:

| Stage | Tokens | Purpose | Mix |
|---|---:|---|---|
| Stage 1 foundation | 1.05T | broad language/code/math base | 65-75% filtered web/edu, 10-15% code, 5-8% math/STEM, 5-10% books/wiki/academic/QA, <=3% synthetic |
| Stage 2 capability | 300B | increase useful skill density | reduce generic web, upsample code/math/STEM/QA, add verified synthetic math/code |
| Stage 3 anneal | 150B | highest quality final distribution | edu/web, code, math/reasoning, long docs, small verified synthetic slice |
| Optional context continuation | 50B-100B | 16k to 32k context | books, code repositories, long web, papers, QA over long documents |

The optional context continuation can be counted inside the 1.5T target by reserving part of Stage 3, or run after the 1.5T base as a 1.55T-1.60T checkpoint.

## 8. Data Requirements

Every document must carry metadata before tokenization:

```text
source
source_version
license
language
domain
quality_score
synthetic_flag
synthetic_model
dedup_cluster
decontam_status
eval_overlap
tokenizer_hash
document_hash
```

Filtering order:

1. Normalize and extract text.
2. Language ID.
3. Domain classification.
4. Exact document dedup.
5. Paragraph and line dedup.
6. MinHash or SimHash near-dedup.
7. Quality filtering, preferably model/classifier assisted.
8. PII, toxicity, and license filters.
9. Eval decontamination.
10. Tokenization.
11. Packed corpus write with source and boundary metadata preserved.

Data source policy:

- Favor DCLM/FineWeb-Edu-style filtered web over raw Common Crawl.
- Keep code data explicit and licensed.
- Keep math/STEM separate so it can be upsampled late.
- Use multilingual data intentionally, not as an accidental web tail.
- Hindi/Indic data must get its own validation slices and tokenizer fertility audit.
- Synthetic data must be tagged, bounded, and eval-filtered.

Synthetic data cap:

```text
Stage 1: <=3%
Stage 2: <=7%
Stage 3/context/reasoning: <=10%
```

Synthetic data is acceptable for:

- textbook-style exposition;
- verified math solutions;
- verified code exercises;
- OCR/PDF cleanup;
- teacher reasoning traces for mid-training or SFT.

Synthetic data is not acceptable as:

- large unverified web replacement;
- recursive self-generated data without fresh real-data anchor;
- untagged corpus content.

### 8.1 Concrete Dataset Portfolio

Pre-2 should use a portfolio, not one monolithic dataset. The mix should be staged and every source must pass license, quality, dedup, decontam, and tokenizer-fertility checks before packing.

Recommended Stage 1 foundation mix:

| Bucket | Target share | Candidate sources | Notes |
|---|---:|---|---|
| High-quality educational web | 35-45% | FineWeb-Edu, DCLM-Baseline, selected FineWeb | Main capability source. Prefer classifier-filtered educational web over raw crawl. |
| General/open diverse web | 10-20% | Dolma, curated FineWeb/FineWeb2 slices | Use to avoid overfitting to exam-style educational prose. |
| Multilingual and Indic | 8-12% | FineWeb2, Indic-specific curated corpora, Wikipedia/Wikisource slices | Hindi/Indic must be explicit, with fertility and held-out evals. |
| Code | 10-15% | The Stack v2 / StarCoder2Data, StarCoder2 extras, permissive GitHub/documentation slices | License and gating are major blockers. Prefer traceable/permissive subsets. |
| Math/STEM | 5-8% | FineMath, OpenWebMath, Proof-Pile-2, MegaMath subsets | Keep separate for late upsampling. |
| Books/reference/wiki/academic | 5-10% | Project Gutenberg, Wikipedia/Wikibooks, arXiv/S2ORC-style permitted slices, Dolma reference subsets | Useful for long-form coherence and factual density. |
| Q&A / StackExchange / documentation | 3-6% | StackExchange, GitHub issues, docs, high-quality forum/documentation sources | Needs formatting and license review. |
| Synthetic | 0-3% | Cosmopedia-style textbooks, verified generated math/code | Tag strictly. Do not let synthetic dominate Stage 1. |

Recommended Stage 2 capability mix:

| Bucket | Target share |
|---|---:|
| Educational/general web | 45-55% |
| Code | 15-20% |
| Math/STEM | 12-18% |
| Q&A/documentation | 8-12% |
| Multilingual/Indic | 8-12% |
| Synthetic verified math/code | 3-7% |

Recommended Stage 3 anneal mix:

| Bucket | Target share |
|---|---:|
| Highest-quality educational web | 35-45% |
| Code/documentation | 20-25% |
| Math/STEM/reasoning | 15-20% |
| Books/reference/long documents | 10-15% |
| Multilingual/Indic | 5-10% |
| Synthetic reasoning/textbook traces | 5-10% |

Dataset source notes:

- FineWeb-Edu is the first-choice educational web base because it is classifier-filtered, released with an ODC-By license, and available at trillion-token scale.
- DCLM-Baseline is the strongest open-data curation reference for web filtering and should be used either directly where possible or as a quality benchmark for our own filtering.
- Dolma is valuable as a transparent, mixed-domain corpus and as a governance reference, but should not be copied blindly; select high-value subsets.
- FineWeb2 is the main candidate for multilingual breadth. It should be sampled by language and quality, not streamed wholesale.
- The Stack v2 / StarCoder2Data is the strongest open code-data lineage, but access, license obligations, PII, opt-out, and attribution requirements must be handled before inclusion.
- FineMath, OpenWebMath, Proof-Pile-2, and MegaMath are late-stage math/STEM candidates. MegaMath includes synthetic and web/code-derived material, so provenance tags are mandatory.
- Cosmopedia-style synthetic data is allowed as a small, tagged component. It is not a substitute for real web/book/code/math data.

Do not use any dataset just because it is large. The correct pre-2 data target is "1.5T useful tokens", not "1.5T available tokens".

### 8.2 Dataset Acceptance Gates

A source enters the pre-2 corpus only if all gates pass:

| Gate | Requirement |
|---|---|
| License | commercial/research posture documented; redistribution constraints known |
| Availability | deterministic snapshot or revision ID |
| Quality | quality score distribution inspected; low-quality tail removable |
| Dedup | exact and near-dedup compatible with global dedup plan |
| Decontam | benchmark overlap report generated |
| PII/safety | source-appropriate PII and unsafe-content filters applied |
| Tokenizer | fertility measured against 64k/96k/131k candidates |
| Metadata | required document metadata fields present |
| Rebuild | corpus shard can be rebuilt byte-identically from manifest |

Dataset rejection triggers:

- unclear license or gated terms not accepted;
- no stable revision or provenance trail;
- high eval contamination that cannot be filtered;
- high duplicate rate after dedup;
- high tokenizer fertility in target languages;
- synthetic content without source-model/prompt metadata.

## 9. Tokenizer Decision

The current 131k SentencePiece tokenizer is defensible, but not automatically final.

Before the 1.5T run, run tokenizer ablations:

| Tokenizer | Purpose |
|---|---|
| 64k | throughput and smaller embedding baseline |
| 96k | middle ground |
| 131k current | continuity and multilingual/code coverage |
| 128k/151k BPE-compatible experiment | optional if aligning with Llama/Qwen-style tooling |

Required tokenizer tests:

- bytes per token by source;
- tokens per word by language;
- code fertility;
- math symbol and digit handling;
- Hindi/Indic fertility;
- training throughput impact;
- downstream proxy eval after 10B-30B tokens.

Hard rule:

Every corpus, checkpoint, teacher artifact, eval run, and scorecard must store the tokenizer hash. Tokenizer identity is a training contract.

## 10. Distillation Decision

The current cross-tokenizer top-K logit plan is disabled for pre-2.

Reason:

Top-K logit KD assumes teacher top-K indices refer to the same vocabulary as the student. That is false for heterogeneous teachers. Clamping or remapping token IDs is not distillation.

Allowed distillation paths:

1. Same-tokenizer logit KD:
   - allowed only when teacher and student tokenizers are byte-for-byte identical;
   - useful if pre-2 adopts a teacher-family tokenizer.

2. Teacher-generated text distillation:
   - generate documents, explanations, math solutions, code examples, and rewrites;
   - tokenize with student tokenizer;
   - train with normal CE;
   - safest pretraining-compatible path.

3. Reasoning/rationale distillation:
   - use teacher reasoning traces for late mid-training or SFT;
   - keep separate from raw pretraining data;
   - tag source model and generation prompt.

4. On-policy distillation:
   - student samples;
   - teacher critiques, rewrites, or scores;
   - suitable after base model is coherent.

5. Research-only cross-tokenizer KD:
   - only with explicit methods such as ALM/ULD;
   - not with ID clipping, heuristic remapping, or vocab-range clamping.

## 11. Software Stack Requirements

Training stack:

| Component | Requirement |
|---|---|
| OS | Linux, not Windows |
| Python | 3.12 preferred |
| PyTorch | 2.12+ unless cluster drivers force a lower version |
| CUDA | CUDA 13.x for newest wheels, or CUDA 12.8 if provider image requires |
| NCCL | provider-tested multi-GPU build |
| Trainer | TorchTitan |
| Parallelism | FSDP2/DTensor first |
| Checkpointing | PyTorch Distributed Checkpointing |
| Precision | BF16 first |
| FP8 | research/performance phase only |
| Attention | native PyTorch SDPA/Flash/FlexAttention path |
| Eval serving | vLLM or SGLang |
| Eval harness | lm-eval and/or lighteval |

Environment rule:

Do not keep production JAX and production PyTorch training in one environment. Use separate containers:

```text
container-data      = corpus build, Rust/Python filters, tokenization
container-train     = PyTorch/TorchTitan/FSDP2
container-eval      = vLLM/SGLang/lm-eval
container-reference = existing JAX/Keras pre-1 code
```

### 11.1 Precision Policy

There are four different "low precision" questions. They should not be mixed.

Training compute precision:

- Baseline pre-2 training should use BF16.
- FP8 training is a performance optimization after the BF16 baseline is stable.
- FP4/NVFP4 training is research-only for this project timeline.

Optimizer precision:

- Start with AdamW using mature TorchTitan/FSDP2 defaults.
- 8-bit optimizer states may be tested to reduce memory, but they are not required for a 1.5B dense model on H100/H200-class hardware.
- Do not combine a new optimizer, FP8 training, and a new architecture in the first main run.

Checkpoint/release precision:

- Keep the canonical training checkpoint in BF16 or a lossless training-state format.
- Export a BF16 SafeTensors checkpoint for eval and downstream conversion.
- Release quantized variants after evaluation, not as the only artifact.

Inference quantization:

| Format | Role |
|---|---|
| BF16 | canonical quality reference |
| FP8 | preferred high-throughput server inference if supported |
| INT8 / SmoothQuant-style | safe production compression target |
| 4-bit AWQ/GPTQ/GGUF | local/edge release artifact after calibration and eval |
| QLoRA/NF4 | fine-tuning method for adapters, not the base pretraining format |

Decision:

Do not train the base model from scratch directly as a 4-bit model. Train in BF16 first, optionally optimize training with FP8 later, then produce 8-bit and 4-bit deployment variants from the trained checkpoint.

Expected model weight sizes:

| Format | Approx weight size for 1.49B params |
|---|---:|
| BF16 | 3.0GB |
| FP8 / INT8 | 1.5GB |
| 4-bit | 0.75GB plus scales/metadata |

Actual runtime memory is higher than weight size because of KV cache, activations, framework overhead, and batching.

## 12. Hardware Requirements

Development:

| Use | Hardware |
|---|---|
| unit tests and CPU data tests | local CPU |
| single-GPU model smoke | RTX 4090, A6000, L40S, A100, H100, or H200 |
| distributed smoke | 2-8 GPUs, same node |

Data preparation:

| Task | Recommended hardware |
|---|---|
| 1T-1.5T corpus build | 128 vCPU, 512GB RAM, fast NVMe |
| global near-dedup at 1T+ | temporary 1TB RAM machine or partitioned external pass |
| object storage | S3/R2-compatible bucket |
| local scratch | 20TB+ NVMe or attached SSD for large builds |

Main dense training:

| Tier | Hardware | Use |
|---|---|---|
| Minimum | 8x H100 80GB or 8x H200 141GB | possible, slower, good for 1.0T-1.5T |
| Recommended | 16x H200 or 16x B200 | practical 1.5T run |
| Strong | 32x H200 or 16-32x B200/GB200 | practical 3T stretch |
| MoE research | 32x H100/H200 minimum | only if using expert parallelism |

Networking:

- Single-node NVLink is enough for the minimum dense run.
- Multi-node training should use 200Gbps or faster InfiniBand.
- Avoid multi-node Ethernet-only training for the main run unless benchmarking proves it is stable and efficient.

Storage:

Packed uint32 tokens:

| Token count | Raw token storage |
|---:|---:|
| 1.0T | 4TB |
| 1.5T | 6TB |
| 3.0T | 12TB |
| 5.0T | 20TB |

Plan for additional metadata, manifests, logs, temporary text, and dedup artifacts. Practical storage should be 2x-4x raw token storage during corpus construction.

Checkpoint storage:

- Dense full training-state checkpoint: approximately 25GB-40GB, depending on optimizer/master-weight format.
- Portable BF16 weights-only checkpoint: approximately 3GB-4GB.
- Save full training checkpoints every 10B-25B tokens.
- Keep last 5 full checkpoints plus milestone checkpoints.
- Export weights-only SafeTensors for eval milestones.

## 13. Training Throughput and Timeline

Throughput estimates for the 1.49B dense model at 8k context:

| Hardware | Planning throughput | 1.5T wall time |
|---|---:|---:|
| 8x H200 | 200k-330k tok/s | 53-87 days |
| 16x H200 | 400k-650k tok/s | 27-43 days |
| 32x H200 | 800k-1.2M tok/s | 14-22 days |
| 8x B200/GB200-class | 350k-600k tok/s | 29-50 days |

These are planning ranges. Actual throughput must be measured after the TorchTitan port with the real packed dataloader.

GPU-hour estimate for 1.5T:

| Hardware | GPU-hours |
|---|---:|
| 8x H200 at 53-87 days | 10k-17k GPU-hours |
| 16x H200 at 27-43 days | 10k-17k GPU-hours |
| 32x H200 at 14-22 days | 11k-17k GPU-hours |

Scaling reduces calendar time more than total GPU-hours.

Calendar plan:

| Phase | Duration | Exit criteria |
|---|---:|---|
| P0 decision freeze | 2-3 days | this doc accepted, model/config frozen |
| P1 TorchTitan skeleton | 1-2 weeks | model forward/loss/checkpoint works on 1 GPU |
| P2 FSDP2 distributed smoke | 1 week | 8-GPU synthetic data run, checkpoint/resume passes |
| P3 packed dataloader port | 1-2 weeks | exact resume by sequence id, tokenizer/data hash checks |
| P4 data pipeline upgrade | 3-5 weeks parallel | 1T+ corpus build feasible, metadata complete |
| P5 proxy runs | 2-3 weeks | 100M-400M models over 10B-50B tokens, mix/tokenizer decision |
| P6 1.5B canary | 1 week | 10B-50B tokens, stable loss, target throughput |
| P7 main 1.5T run | 2-12 weeks depending hardware | no fatal instability, eval curves acceptable |
| P8 context continuation | 3-10 days | 16k/32k eval passes |
| P9 final eval/governance | 1 week | scorecard, model card, data card, release decision |

Practical total timeline:

- With 8x H200: approximately 14-18 weeks.
- With 16x H200/B200: approximately 10-13 weeks.
- With 32x H200 or equivalent: approximately 8-10 weeks.

## 14. Go/No-Go Gates

Pre-run gates:

- tokenizer ablation complete;
- data license register complete;
- decontam index complete;
- packed dataloader exact resume proven;
- FSDP2 checkpoint restore across process count proven;
- eval harness can run real model predictions;
- no mock release scorecard paths for release decisions.

10B-token gate:

- no recurring NaNs;
- loss decreases smoothly;
- checkpoint/resume exactness proven;
- source-bucket losses logged;
- throughput at least 70 percent of planning target.

50B-token gate:

- proxy evals improving;
- no source bucket collapse;
- no unexpected memorization spike;
- data order replayable;
- no corruption in packed corpus manifests.

300B-token gate:

- compare against proxy baselines;
- decide whether to continue to 1T/1.5T;
- decide whether Stage 2 mix needs more code/math/STEM.

1.0T-token gate:

- internal checkpoint candidate;
- run full eval;
- continue to 1.5T unless eval/loss curves are clearly saturated or budget is exhausted.

1.5T-token gate:

- pre-2 base release candidate;
- final scorecard;
- governance docs;
- contamination report;
- release or continue-to-3T decision.

## 15. Evaluation Plan

Perplexity:

- Paloma;
- held-out source buckets;
- held-out Hindi/Indic validation;
- held-out code;
- held-out math/STEM;
- long-document validation after context continuation.

Task eval:

- MMLU or MMLU-Pro;
- ARC-Easy and ARC-Challenge;
- HellaSwag;
- Winogrande;
- BoolQ;
- GSM8K;
- MATH-500;
- HumanEval+;
- MBPP+;
- Belebele;
- MILU or equivalent Indic eval if access is available;
- RULER/LongBench-style long-context eval after context continuation.

Baselines:

- Llama 3.2 1B;
- Qwen3 0.6B and Qwen3 1.7B;
- SmolLM2 1.7B;
- OLMo 2 1B where comparable.

Expected result range:

- At 1.0T: credible internal checkpoint, not a strong public release unless data quality is excellent.
- At 1.5T: realistic public base candidate if data quality and post-training are strong.
- At 3.0T: stronger chance of beating older 1B-class public baselines, but not guaranteed.

Do not promise Llama 3.2 1B parity only from parameter count. Llama 3.2 1B used up to 9T tokens and same-family distillation.

## 16. MoE Research Branch

Only pursue after the dense pre-2 baseline is running.

Target:

```text
myllm-pre2-moe-1b-active-7b-total-research
```

High-level requirements:

- 1B-1.3B active parameters;
- 6B-8B total resident parameters;
- top-k routed experts;
- expert parallelism;
- router load balancing;
- grouped GEMM;
- token dispatcher;
- MoE-aware checkpointing;
- MoE-aware eval and serving path.

Recommended stack:

- Megatron-Core/NeMo Megatron Bridge if using NVIDIA cluster discipline;
- TorchTitan MoE if it satisfies needed expert-parallel features at implementation time.

Do not implement MoE in the current custom Keras/JAX stack.

## 17. Risks

Top risks:

1. Data quality is below the model/training ambition.
2. TorchTitan migration takes longer than expected.
3. Packed dataloader resume semantics are wrong.
4. Tokenizer choice wastes too many parameters or hurts code/Indic fertility.
5. Eval scorecard remains scaffolded instead of real.
6. Cross-tokenizer distillation accidentally re-enters the plan.
7. 8-GPU training is too slow for 1.5T.
8. Governance docs drift from live configs.

Mitigations:

- run proxy data/tokenizer experiments before the main run;
- keep dense architecture conservative;
- use mature PyTorch distributed primitives;
- keep JAX as reference only;
- make data manifests and tokenizer hashes mandatory;
- require real eval before any release claim.

## 18. Required Repo Changes

Docs:

- [x] add this pre-2 plan;
- [x] mark the remaining docs as current pre-1/pre-2 artifacts where needed;
- [x] add a pre-2 architecture decision record;
- [x] add a pre-2 stack migration plan;
- [x] convert the governance data card into a pre-2 release scaffold.

Configs:

- [x] `configs/pre2_dense_1_5b.yaml`;
- [x] `configs/pre2_dense_proxy_400m.yaml`;
- [x] `configs/pre2_moe_1b_active_research.yaml`;
- [x] `configs/data/pre2_mix_stage1.yaml`;
- [x] `configs/data/pre2_mix_stage2.yaml`;
- [x] `configs/data/pre2_mix_anneal.yaml`.

Code:

- [x] pre-2 config schema and validation CLI;
- [x] minimal PyTorch dense model adapter for shape/loss tests;
- [x] fail-closed top-K KD guard for pre-2 batches;
- [x] CPU/1-GPU synthetic pre-2 training smoke entrypoint;
- [x] first packed-corpus PyTorch dataloader skeleton;
- [x] single-process checkpoint payload for model, optimizer, scheduler, data cursor, and RNG;
- [ ] TorchTitan trainer integration;
- [ ] TorchTitan packed-corpus dataloader;
- [ ] FSDP2 checkpoint/resume harness;
- [ ] real eval predict function;
- [ ] SafeTensors export;
- [ ] remove or fail-close heterogeneous-tokenizer logit KD.

Tests:

- [x] config and parameter-count tests for pre-2 planning configs;
- [x] minimal model shape/loss tests;
- [x] top-K KD guard tests;
- [x] packed dataloader sequence-id resume skeleton tests;
- [x] real packed-corpus loader fixture test;
- [ ] full packed-corpus exact resume tests over checkpoint restore;
- [ ] distributed checkpoint round-trip tests;
- [ ] 1 GPU, 2 GPU, 8 GPU smoke tests;
- [ ] tokenizer hash mismatch fail-closed tests;
- [ ] eval scorecard non-mock test.

## 19. Source Basis

Primary sources consulted for this decision:

- Qwen3 Technical Report: https://arxiv.org/abs/2505.09388
- DeepSeek-V3 Technical Report: https://arxiv.org/abs/2412.19437
- OLMoE model card: https://huggingface.co/allenai/OLMoE-1B-7B-0125
- SmolLM3 technical blog: https://huggingface.co/blog/smollm3
- Llama 3.2 1B model card: https://huggingface.co/meta-llama/Llama-3.2-1B
- PyTorch FSDP2 docs: https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html
- TorchTitan repository: https://github.com/pytorch/torchtitan
- PyTorch 2.12 release: https://pytorch.org/blog/pytorch-2-12-release-blog/
- Megatron-Core MoE docs: https://docs.nvidia.com/megatron-core/developer-guide/0.17.0/user-guide/features/moe.html
- Cross-tokenizer distillation via ALM: https://arxiv.org/abs/2503.20083
- Multi-token prediction: https://arxiv.org/abs/2404.19737
- DCLM: https://arxiv.org/abs/2406.11794
- Dolma: https://arxiv.org/abs/2402.00159
