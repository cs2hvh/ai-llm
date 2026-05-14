# Pre-2 Independent Research Synthesis

Date: 2026-05-14
Branch: beta-dev
Scope: independent challenge review of the pre-2 plan, architecture, stack, data, precision, governance, evaluation, hardware, serving, and current repo evidence.

## Executive Decision

The current pre-2 plan should be revised from "final dense decision" to:

> Dense 1.485B is the default production baseline pending falsification. MoE, FP8, Muon, tokenizer alternatives, data mixture, and long-context choices must be tested through defined POCs before the 1.5T main run is approved.

Dense remains the lowest-risk baseline for a first enterprise-grade 1B-class run, but it is not proven to be the highest-quality path. If the real business goal is best quality at about 1B active parameters and the project can tolerate 6B-8B resident parameters, an OLMoE-style 1B-active / 7B-total MoE branch deserves a serious POC before the dense main run is frozen.

The biggest immediate correction is status honesty: the repo now has strong pre-2 planning and useful smoke scaffolding, but not yet a production TorchTitan/FSDP2/DCP training system, not yet a release-grade eval system, and not yet an enterprise-buildable data corpus.

## Research Coverage

Completed independent tracks:

- Architecture: dense vs MoE, GQA, RoPE/NoPE/YaRN, QK norm, z-loss, scaling.
- Training stack: PyTorch/TorchTitan/FSDP2 vs Megatron-Core, NeMo, DeepSpeed, MaxText/JAX.
- Data and governance: dataset portfolio, licensing, synthetic accounting, decontamination.
- Tokenizer: SentencePiece-Unigram/BPE, vocab size, byte fallback, Indic/code fertility.
- Precision: BF16, FP8, INT8/INT4, QAT/PTQ, stability.
- Evaluation: benchmark matrix, contamination, scorecard hard gates.
- Checkpoint/resume: DCP, exact replay, process-count restore, data cursor.
- Hardware: H100/H200/B200 topology, MFU assumptions, timeline.
- Serving/export: HF, vLLM, TensorRT-LLM, SGLang, GGUF, ONNX.
- Optimizer/schedule: AdamW/WSD, Muon, Sophia, Lion, Adafactor.
- Codebase audit: implementation-vs-doc overclaim checks.
- POCs: model runtime smoke and data-resume stress scripts under `tmp/agent-pocs`.

One governance agent failed due context-window exhaustion; governance synthesis below is based on primary sources: NIST AI RMF, ISO/IEC 42001, EU AI Act GPAI guidance, SPDX 3.0.1, dataset license/source documents, and model-card/data-card literature.

## Current Repo Evidence

What is proven locally:

- Pre-2 config/model/guard/loader/checkpoint smoke tests pass when run with `PYTHONPATH=src`.
- Packed-corpus tests pass.
- Current dense config estimates about `1.485B` parameters, within about `0.34%` of the target.
- Tiny PyTorch pre-2 model forward/backward is finite and deterministic on CPU.
- Throwaway model POC showed AdamW state is populated after one step.
- Throwaway data-resume POC showed monotonic sequence IDs, shard-boundary reads, tokenizer hash rejection, and exact replay from a sequence cursor.

What is not yet proven:

- Full 1.485B allocation, GPU memory, MFU, or throughput.
- TorchTitan/FSDP2/DTensor training.
- PyTorch Distributed Checkpointing and process-count restore.
- BF16/FP8/INT8/INT4 training/export behavior.
- Real checkpoint restore into training and exact continuation.
- Release-grade evaluation with real predictions.
- Enterprise-governed 1.5T corpus build.
- Serving parity through HF/vLLM/TensorRT-LLM/SGLang/GGUF/ONNX.

Known overclaims to fix before enterprise review:

- README quickstart does not work without `PYTHONPATH=src` or editable install.
- Docs say PyTorch 2.12+ but `requirements-gpu.txt` still pins `torch==2.7.1`.
- BF16 is configured but the smoke trainer does not enforce BF16/autocast.
- `loss_mask` is emitted by the pre-2 loader but not consumed by model loss.
- Checkpointing is single-process `torch.save`, not DCP.
- Legacy heterogeneous top-K KD can still activate outside `myllm_pre2`.
- Release scorecard still has mock/non-empty-output scoring paths.
- Governance docs are scaffolds, not generated release evidence.

## Architecture Decision

### Baseline

Use a dense decoder-only Transformer as the default baseline:

- Parameter target: current config about `1.485B`.
- Training target: `1.5T` planned tokens, `1T` hard minimum, `3T` serious release stretch if data and budget hold.
- Components: tied embeddings, RMSNorm/pre-norm, GQA, RoPE-family position encoding, QK norm, SwiGLU, CE + z-loss.
- Precision: BF16 baseline.
- Context: 8k initial config is acceptable for pre-2 bring-up; 16k/32k must be gated by long-context POCs.

Rationale: dense is simpler to train, debug, checkpoint, evaluate, and serve. Current small-model practice still supports strong dense baselines: Qwen3 dense small models, Llama 3.2 1B, Gemma 3 1B, and SmolLM2/SmolLM3 all reinforce dense/GQA/overtraining as a credible path.

### MoE Branch

Do not build a 1B-total MoE; that is capacity fragmentation plus routing complexity.

Do run a serious MoE POC if active-parameter efficiency matters:

- Proxy: dense 400M-600M vs MoE 400M active / 2B-3B total for 30B-50B tokens.
- Canary: dense 1.485B vs MoE about 1B active / 6B-8B total for 100B tokens.
- Promote MoE only if it wins eval-per-GPU-hour by a material margin, has stable routing/load balance, and passes checkpoint/serving tests.
- Prefer Megatron-Core/NeMo for MoE unless TorchTitan proves expert parallelism, grouped GEMM, distributed checkpoint, and serving parity.

### Required Architecture POCs

1. Dense vs MoE proxy with same data order and eval.
2. Attention ablation: MHA, GQA 8KV, GQA 4KV, MQA.
3. Position ablation: RoPE base variants, YaRN, partial-NoPE proxy.
4. QK norm order and stability: QK norm before/after RoPE, z-loss sweep, optional logit softcap.
5. Scaling shape: 750M/200B, 1.5B/100B, 3B/50B iso-FLOP comparison.
6. Long-context continuation: 25B/50B/100B/200B at 16k/32k with RULER/LongBench.

## Training Stack

Ranked stack decision:

1. PyTorch 2.12.x + TorchTitan + FSDP2/DTensor + DCP: selected candidate for dense baseline.
2. Megatron-Core: fallback for throughput and default for serious MoE.
3. NeMo Megatron Bridge: use if NVIDIA container/recipe discipline is desired.
4. DeepSpeed ZeRO-3: tactical comparator, not primary greenfield path.
5. MaxText/JAX/Pax: credible on TPU, but wrong direction unless the PyTorch migration is abandoned.
6. Accelerate/Lightning Fabric/torchtune/custom trainer: useful for POCs or post-training, not main pretraining.

Required stack corrections:

- Pin exact versions: PyTorch 2.12.x, CUDA wheel, driver, NCCL, TorchTitan commit, torchao, container digest.
- Change wording from "production path" to "selected candidate pending POCs."
- Add `src/myllm_pre2/torchtitan/` adapter rather than growing the smoke trainer.
- Replace single-process checkpoint path with DCP app-state wrapper.
- Add 1/2/8 GPU distributed tests and process-count restore tests.
- Run a stack bakeoff: TorchTitan vs Megatron-Core vs DeepSpeed on a 100M-400M proxy with same packed data.

## Data Plan

The staged shape is good, but current configs are bucket-level planning metadata, not an enterprise-buildable corpus.

Revised 1.5T target mix:

| Area | Aggregate Target | Enterprise stance |
|---|---:|---|
| Filtered English edu/general web | 50-54% | FineWeb-Edu primary; selected FineWeb/FineWeb2/Dolma only after global dedup and license review. |
| Code | 13-15% | The Stack v2 or similar only with SPDX allowlist, provenance, SWHID, opt-out refresh, no unknown-license code. |
| Math/STEM | 8-10% | FineMath/OpenWebMath core; MegaMath only with web/code/synthetic strata preserved. |
| Multilingual/Indic | 8-10% | FineWeb2 language-script slices plus Sangraha Verified/Unverified; synthetic translations counted separately. |
| Books/reference/wiki/QA/docs | 8-10% | Use only sources with explicit redistribution/training posture and attribution manifests. |
| Synthetic | 2-4% global | Count all synthetic-origin content, including translations, romanization, generated code/math, teacher traces. |

Required data rule: Common Crawl derivatives are not independent reservoirs. FineWeb, FineWeb-Edu, FineWeb2, DCLM, Dolma, OpenWebMath, FineMath, MegaMath, and similar sources must be globally deduplicated and decontaminated after final normalization.

Required manifests:

- `source_registry`: dataset id, owner, URL, revision, acquisition date, license, terms, allowed use.
- `license_manifest`: per-source/per-document license, attribution, share-alike, opt-out status.
- `document_manifest`: URL/SWHID, language/script, domain, token count, hashes, crawl date, quality score.
- `filter_manifest`: filter versions, thresholds, model hashes, dropped-count reasons.
- `dedup_manifest`: exact hash, paragraph hash, near-dedup cluster id, retain/drop decision.
- `decontam_manifest`: benchmark versions, method, overlap ids, removal decision.
- `synthetic_manifest`: source model, prompt/template hash, generation date, verifier, parent ids.
- `tokenizer_manifest`: tokenizer hash, fertility by source/language, byte fallback rate.
- `packing_manifest`: shard id, token count, source proportions, document boundaries, checksum.
- `deletion_manifest`: opt-outs, takedowns, rebuild impact.

## Tokenizer

Keep the current `131,075` runtime-vocab SentencePiece-Unigram artifact with byte fallback as provisional default, but freeze only after scorecards.

Do not run a blind 32k vs 128k ablation. Use:

- SP-Unigram: 64k, 96k, current 131075 artifact.
- SP-BPE: 64k, 96k, current 131075-size target.
- Optional Qwen-style byte-level BPE control around 151k, only if parameter budget can absorb it.
- 32k only as a tokenizer-only floor test unless it passes Indic/code gates.

Score by bits per byte, not per-token CE, because token counts differ.

Required metrics:

- bytes/token, tokens/1k bytes, tokens/word, tokens/grapheme for Indic/CJK.
- byte fallback rate, unknown-token rate, special-token collision rate.
- p50/p90/p99 document lengths under 8k/16k/32k.
- code metrics: indentation/newline handling, long identifier splitting, digit splitting, JSON/YAML/Markdown fertility.
- throughput: train tokens/sec, real bytes/sec, step time, softmax share.
- proxy quality: source-bucket bits/byte, Hindi/Indic heldout, code heldout, math heldout.

Important correction: current PyTorch parameter estimate omits final RMSNorm parameters. The delta is small, but parameter counts should be labeled approximate until the estimator is reconciled with the model.

## Precision And Quantization

Decision:

- Train BF16 first.
- FP8 is a gated training ablation, not just an inference/export feature.
- Do not train the base model in INT8/INT4.
- INT8/INT4 are deployment/export paths plus optional QAT recovery fine-tune if PTQ fails quality gates.
- QLoRA is not evidence that full 4-bit base pretraining is safe; it trains adapters through a frozen quantized base.

Required precision POCs:

1. BF16 parity on GPU: FP32 vs BF16 loss/grad norms for 200 steps.
2. Stability sweep: QK norm on/off, z-loss `{0,1e-5,1e-4,1e-3}`, logit softcap `{none,30}`.
3. Activation checkpointing parity and peak-memory measurement.
4. FP8 shadow run with TorchTitan/torchao or Transformer Engine for 1k-10k proxy steps.
5. Quant export ladder: BF16, FP8/W8A8, INT8, AWQ/GPTQ INT4, GGUF Q8/Q5/Q4.

## Optimizer And Schedule

Keep AdamW + WSD as the control.

Strongest challenger:

- Muon on hidden 2D matrix weights plus AdamW on embeddings, norms, biases, and heads.

Secondary challengers:

- Sophia-G/H: high upside, higher integration risk.
- Lion: smoke only unless it wins clearly.
- Adafactor: memory fallback, not first choice if AdamW states fit.

Adoption gate: replace AdamW only if challenger reaches the same validation loss with at least 10% lower wall-clock/FLOPs, or gives a stable final-loss gain at equal compute, without worse spikes/NaNs across seeds.

## Hardware And Timeline

Compute for the current dense target:

`6 * 1.485e9 * 1.5e12 = 1.3365e22` training FLOPs before overhead, evals, retries, checkpointing, and data work.

Recommendations:

- Start with an 8-GPU POC on a real HGX/DGX-class node.
- Use 16x H100/H200 BF16 for the main dense run if calendar matters.
- Use B200 only if immediately available and price/performance wins in the POC.
- Use 32x only if it gives at least 3.0x 8-GPU throughput; 1B-class models can underutilize large clusters.
- Avoid weak PCIe/100GbE setups for the main run.
- Use local NVMe cache plus durable object/Lustre/FSx storage for checkpoints and artifacts.

Required throughput gates:

| Gate | Pass target |
|---|---:|
| 8x H100/H200 BF16 synthetic data | prove sustained tokens/sec and MFU; establish baseline |
| 8x real packed data | no dataloader stalls; within 15% of synthetic throughput |
| 16x BF16 | at least 1.7x 8-GPU throughput |
| 32x BF16 | at least 3.0x 8-GPU throughput or reject |
| FP8 | at least 1.6x BF16 throughput with loss parity |
| checkpoint save | less than 3 min pause |
| checkpoint restore | less than 20 min |
| burn-in | 12-24h without NCCL, dataloader, or checkpoint instability |

Timeline:

- 8x H100/H200 BF16: likely 8-11 weeks end to end if systems are immature.
- 16x H100/H200 BF16: likely 6-8 weeks end to end after data/legal/eval are ready.
- 8x B200 or FP8 path: possibly 4-7 weeks, only after POC validation.
- On-prem new hardware: add 12-30+ weeks before training.

## Checkpointing And Resume

Current single-process checkpoint is smoke scaffolding only.

Required architecture:

- Use PyTorch Distributed Checkpointing as canonical training checkpoint.
- Implement TorchTitan-style `AppState` containing model, optimizer, scheduler, train state, dataloader state, RNG streams, manifest hashes.
- Store world-size independent global cursor and deterministic mapping from optimizer step to sequence IDs.
- Checkpoint only on optimizer-step boundaries unless partial accumulation state is persisted.
- Maintain one outstanding async save, keep previous complete checkpoint, write latest pointer last.
- Same-topology restart should be bitwise exact.
- Cross-process-count restore should restore state and data order exactly at load time; subsequent training may be numerically equivalent rather than bitwise identical due to collective order.

Required tests:

- Single-process exact continuation: uninterrupted N steps vs K + save/load + N-K.
- DCP same-world resume 2x2.
- DCP reshard 1 to 2 and 2 to 1.
- Data plan world-size invariant: no duplicates/gaps, same canonical global batch.
- Reject partial DCP checkpoint after killed rank.
- RNG logical-stream restore.
- Forbid mid-accum checkpoint without state.

## Evaluation And Release Gates

The scorecard must become fail-closed and real-prediction based.

Evaluation matrix:

| Track | Required evals | Gate |
|---|---|---|
| Reproducibility | lm-eval or LightEval plus native scorecard | model hash, tokenizer hash, dataset revisions, prompt, decoding, seed, harness commit, CI |
| Perplexity | Paloma, held-out source buckets, code/math/Indic buckets | no unexplained >5% regression; byte-normalized reporting |
| General | MMLU-Pro, ARC, HellaSwag, WinoGrande, BoolQ, BBH | full run, no sample caps for release |
| Math | GSM8K, MATH-500, MGSM | exact-answer scoring; per-language MGSM |
| Code | HumanEval+, MBPP+, LiveCodeBench | sandboxed execution; temporal split after corpus cutoff |
| Multilingual/Indic | MMLU-ProX, Belebele, MILU, MGSM, fertility | Hindi plus target Indic languages separately |
| Long context | RULER, LongBench, position-wise PPL | claim only lengths that pass effective-context threshold |
| Safety/privacy | RealToxicityPrompts, BBQ, SafetyBench/HarmBench, canary extraction, PII audit | disclose for base; block public release on unresolved PII |
| Contamination | static n-gram, fuzzy, code/test-case scan, temporal cutoff | zero unresolved benchmark leakage |
| Governance | model card, data card, eval card, risk card | generated from artifacts, not hand-filled planning text |

Public release should require:

- reproducibility hard gates pass;
- contamination hard gates pass;
- no safety/privacy blocker;
- composite quality at least 85% of median reference baseline across freshly re-run 1B/1.7B comparators;
- complete model/data/eval/risk cards.

Below that quality threshold, release only internally.

## Serving And Export

SafeTensors + INT8/INT4 is not a serving strategy. It is a packaging and compression component.

Required serving matrix:

| Target | Role |
|---|---|
| HF Transformers | canonical correctness, eval, quant calibration, parity |
| vLLM | default GPU serving POC: OpenAI server, PagedAttention, batching, prefix cache, FP8 KV |
| TensorRT-LLM | NVIDIA-optimized path: IFB, paged attention, FP8/INT8/INT4/FP4, genai-perf |
| SGLang | prefix-heavy, structured output, agentic workflows |
| llama.cpp/GGUF | local/edge/CPU/desktop export |
| ONNX Runtime GenAI | Windows/DirectML/portable embedding |

Required export steps:

1. Lock HF-compatible architecture and tensor naming.
2. Export canonical BF16 SafeTensors with config/tokenizer/generation config.
3. Quantize only after parity gates.
4. Run KV-cache quantization experiments for 8k/16k/32k.
5. Add serving harness, workload JSONL, TTFT/TPOT/throughput metrics, GPU memory logging.

## Governance And Compliance

Governance is a release blocker, not a paperwork phase.

Minimum framework mapping:

- NIST AI RMF: map, measure, manage, govern risks across lifecycle.
- ISO/IEC 42001: maintain an AI management system with roles, controls, monitoring, and continuous improvement.
- EU AI Act GPAI obligations if placed in EU market: technical documentation, copyright policy, and public training-data summary.
- SPDX 3.0.1: use SBOM/BOM concepts for software, datasets, model artifacts, and training pipeline provenance.

Required artifacts:

- AI risk register and risk acceptance log.
- Model BOM: code commit, dependencies, containers, training stack, tokenizer, datasets, evals, weights, quant variants.
- Dataset BOM: source registry, licenses, transformations, filtering, dedup, decontam, synthetic provenance.
- Copyright and license policy.
- PII handling policy and deletion/takedown procedure.
- Security and secrets policy for code data.
- Model card, data card, eval card, risk card.
- Release notes with limitations and prohibited or unsupported uses.
- Incident response and rollback plan.
- Audit log tying every released weight to config, corpus manifest, checkpoint, eval report, and approval.

## POC Approval Ladder

### P0 - Repo Truthfulness And Packaging

Exit criteria:

- README quickstart works in a clean env.
- Docs label target vs implemented status.
- PyTorch dependency pins align with pre-2 plan.
- No Word lock files or pycache artifacts.
- Legacy heterogeneous top-K KD cannot activate from pre-2 entrypoints.

### P1 - Local Correctness

Exit criteria:

- Model runtime POC is promoted into tests.
- Data-resume POC gaps are closed.
- `loss_mask` is consumed by pre-2 loss.
- Exact single-process checkpoint resume test passes.
- Tokenizer hash and corpus manifest hash are fail-closed.

### P2 - Distributed Runtime

Exit criteria:

- TorchTitan adapter parity with local model.
- FSDP2/DCP 1/2/8 GPU smoke tests.
- DCP reshard restore tests.
- BF16 GPU parity and stability sweep.
- 12-24h burn-in on real packed data.

### P3 - Data And Tokenizer Gate

Exit criteria:

- 30B-50B corpus POC with 100% source/license/manifest coverage.
- Global dedup/decontam report.
- Tokenizer scorecard selects production tokenizer.
- Synthetic accounting includes translations, generated math/code, teacher traces.
- Indic/code/math heldouts and fertility pass.

### P4 - Model Selection

Exit criteria:

- 100M-400M proxy stack bakeoff.
- Dense vs MoE proxy.
- AdamW/WSD vs Muon challenger.
- Precision BF16 vs FP8 shadow run.
- Eval harness real-prediction POC.

### P5 - Main Run Authorization

Exit criteria:

- Hardware throughput gate passes.
- Training stack, data, tokenizer, eval, governance, and checkpoint gates are green.
- Compute and cost envelope approved.
- Public vs internal release threshold defined.

### P6 - Post-Training, Export, Release

Exit criteria:

- HF BF16 parity.
- Quant export ladder.
- vLLM/TRT/SGLang/GGUF/ONNX POCs.
- Full eval/decontam/risk reports.
- Generated model/data/eval/risk cards.

## Immediate Repo Changes Recommended

1. Add `docs/PRE2_STATUS_MATRIX.md` separating implemented, tested, planned, blocked.
2. Fix pre-2 packaging and dependency pins.
3. Add pre-2 extras for PyTorch/TorchTitan/DCP once selected.
4. Reword README and DOCX language that implies production stack is already implemented.
5. Promote throwaway POCs into tracked tests after review.
6. Implement `loss_mask` in pre-2 loss.
7. Add data/corpus manifest hash validation to pre-2 loader and checkpoints.
8. Implement exact single-process resume before DCP.
9. Build DCP app-state wrapper.
10. Replace mock eval scorecard with real harness ingestion.
11. Add tokenizer scorecard scripts.
12. Add governance artifact generators from immutable manifests.

## Source Anchors

Primary technical and governance sources used or cited by the review:

- Qwen3 technical report: https://arxiv.org/abs/2505.09388
- OLMoE: https://arxiv.org/abs/2409.02060
- Llama 3.2 1B model card: https://huggingface.co/meta-llama/Llama-3.2-1B
- SmolLM2: https://arxiv.org/abs/2502.02737
- SmolLM3 recipe: https://huggingface.co/blog/smollm3
- Gemma 3 model card: https://ai.google.dev/gemma/docs/core/model_card_3
- PyTorch 2.12 release: https://pytorch.org/blog/pytorch-2-12-release-blog/
- TorchTitan: https://github.com/pytorch/torchtitan
- PyTorch FSDP2: https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html
- PyTorch DCP: https://docs.pytorch.org/docs/2.9/distributed.checkpoint.html
- Megatron-Core: https://developer.nvidia.com/megatron-core
- FineWeb/FineWeb-Edu: https://arxiv.org/abs/2406.17557
- DCLM: https://arxiv.org/abs/2406.11794
- Dolma: https://arxiv.org/abs/2402.00159
- The Stack v2 / StarCoder2: https://arxiv.org/abs/2402.19173
- OpenWebMath: https://arxiv.org/abs/2310.06786
- SentencePiece: https://arxiv.org/abs/1808.06226
- QLoRA: https://arxiv.org/abs/2305.14314
- SmoothQuant: https://arxiv.org/abs/2211.10438
- AWQ: https://arxiv.org/abs/2306.00978
- lm-eval-harness: https://github.com/EleutherAI/lm-evaluation-harness
- Paloma: https://arxiv.org/abs/2312.10523
- RULER: https://github.com/NVIDIA/RULER
- vLLM docs: https://docs.vllm.ai/
- TensorRT-LLM docs: https://nvidia.github.io/TensorRT-LLM/
- SGLang docs: https://docs.sglang.ai/
- llama.cpp / GGUF: https://github.com/ggml-org/llama.cpp
- ONNX Runtime GenAI: https://onnxruntime.ai/docs/genai/
- NIST AI RMF: https://www.nist.gov/itl/ai-risk-management-framework
- ISO/IEC 42001: https://www.iso.org/standard/42001
- EU AI Act GPAI Q&A: https://digital-strategy.ec.europa.eu/en/faqs/general-purpose-ai-models-ai-act-questions-answers
- SPDX 3.0.1: https://spdx.github.io/spdx-spec/v3.0.1/
