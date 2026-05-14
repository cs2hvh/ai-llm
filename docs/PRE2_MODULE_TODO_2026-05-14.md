# Pre-2 Module TODO Backlog

Date: 2026-05-14
Branch: `beta-dev`
Goal: build a research-grade path from the current pre-1 repository to a pre-2 1.5B-class dense baseline, trained on at least 1T internal tokens and targeted at 1.5T+ total tokens, with MoE treated as a separate research branch.

## Priority Legend

- P0: correctness, reproducibility, or truthfulness blocker. Do before scale work.
- P1: minimum viable pre-2 training system on 1-8 GPUs.
- P2: scale-out and production hardening.
- P3: post-base release, serving, and governance polish.

## Active Implementation Slice

These items are started first because they unblock reliable development:

| Status | Priority | Module | Task | Exit criteria |
| --- | --- | --- | --- | --- |
| Done | P0 | Repo environment | Make local install and README quickstart match the `src/` package layout. | Fresh venv can import `myllm_pre2` after documented install. |
| Done | P0 | Pre-2 loss | Consume packed-corpus `loss_mask` in the PyTorch dense model. | Masked tokens do not affect CE or z-loss; unit test covers masked labels. |
| Done | P0 | Dependency posture | Separate pre-1 GPU pod recipe from pre-2 PyTorch/TorchTitan dependency plan. | README and package extras state which deps are runnable now versus pending lock. |
| Done | P1 | Training runtime | Add a single-process packed-corpus smoke path. | Tiny real packed-corpus fixture trains one step through model, labels, and loss mask. |
| Done | P1 | Precision | Add BF16/autocast path to the smoke trainer. | `--precision bf16` and `--precision config` run finite CPU smoke steps. |
| Done | P1 | Training telemetry | Add grad clipping and basic step telemetry to smoke trainer. | Smoke output reports tokens, LR, grad norm, grad clip, and peak CUDA memory when applicable. |
| Done | P1 | Checkpoint/resume | Add exact single-process checkpoint restore and trainer resume smoke. | Restored model, optimizer, RNG, and data cursor match uninterrupted training in tests. |
| Done | P0 | Release planning | Codify v0.5/v1 launch gates and the model/token decision. | `docs/PRE2_RELEASE_PLAN.md` defines stage scope and hard gates. |
| Done | P0 | Compute-limited POC ladder | Add 110M canary and 250M POC configs before 400M/1.5B. | Config checker and tests validate the lower-cost ladder. |
| Done | P1 | Eval bridge | Add a tiny real next-token eval path for pre-2 checkpoints. | Smoke checkpoints load into a model and produce greedy next-token predictions. |
| Done | P0 | Source registry | Add pre-2 source registry and license gate. | Registry validates sources and fails closed for unapproved training stages. |
| Done | P0 | v0.5 data gate | Pin and approve the two-source v0.5 POC mix. | `scripts/pre2_source_registry_check.py --require-stage poc` passes for FineWeb-Edu and OpenWebMath. |
| Done | P0 | v0.5 readiness CLI | Add a launch gate for source approval, mix consistency, tokenizer presence, and storage estimates. | `scripts/pre2_v0_5_readiness.py` blocks on missing tokenizer and passes with a tokenizer artifact. |
| Done | P1 | v0.5 corpus build plan | Add build-time HF streaming config and command emitter. | `scripts/pre2_v0_5_build_commands.py` emits pinned per-source `build_packed_corpus.py` commands. |
| Done | P1 | CPU data-prep VM | Bootstrap Linux VM, storage, dependencies, R2 artifacts, and readiness gate. | VM has 64 CPU cores, 83GiB RAM, 800GiB root disk, artifacts staged, and pre-2 tests passing. |
| Done | P0 | Tokenizer runtime contract | Correct pre-2 configs from rounded 131,072 planning vocab to actual 131,075 runtime vocab. | Parameter math and tests use max tokenizer ID + 1. |
| Done | P0 | Verification | Run focused pre-2 tests and config smoke. | Pre-2 tests pass and smoke train returns finite loss. |

## 0. Program Controls

| Priority | Task | Notes | Exit criteria |
| --- | --- | --- | --- |
| P0 | Maintain a decision log for architecture, data, precision, tokenizer, and serving choices. | ADRs must distinguish selected defaults from research branches. | Every major choice has evidence, alternatives, and reversal trigger. |
| P0 | Track pre-1 versus pre-2 scope explicitly. | Avoid claiming pre-2 production readiness while only smoke scaffolds exist. | README, plans, and configs use consistent status language. |
| P1 | Define milestone gates: POC, proxy, pilot, base, post-train, release. | Each gate should have compute budget, data budget, eval gates, and rollback criteria. | No training stage starts without a signed gate checklist. |
| P1 | Add risk register. | Include data license risk, GPU supply risk, eval leakage, resume failure, and model misuse risk. | Risks have owner, mitigation, and current status. |

## 1. Repository, Packaging, And Environment

| Priority | Task | Notes | Exit criteria |
| --- | --- | --- | --- |
| P0 | Make `pip install -e .` the default local workflow. | Current scripts import from `src`; plain `requirements.txt` is not enough. | README quickstart imports `myllm` and `myllm_pre2` without `PYTHONPATH` hacks. |
| P0 | Add a pre-2 optional dependency group. | Include PyTorch for smoke/adapter work; leave TorchTitan commit pin for the container lock. | `pip install -e ".[dev,pre2]"` supports pre-2 tests. |
| P0 | Clarify `requirements-gpu.txt` ownership. | It currently reflects pre-1 single-venv JAX + Torch audit constraints, not the final pre-2 stack. | Docs do not imply the old GPU file is the TorchTitan 2.12+ training image. |
| P1 | Add a pre-2 container lock. | Pin Python, PyTorch, CUDA wheel/index, NCCL, TorchTitan commit, torchao, flash-attn path, and image digest. | Container reproduces on a clean GPU node. |
| P1 | Add environment smoke matrix. | CPU smoke, single GPU, 8 GPU, and data-prep worker. | CI/manual scripts report import, CUDA, NCCL, PyTorch, tokenizer, and packed-corpus health. |
| P2 | Adopt a real lockfile. | Use `uv.lock` or pip-compile outputs per environment. | Production runs never install from floating broad ranges. |

## 2. Architecture And Dense Model

| Priority | Task | Notes | Exit criteria |
| --- | --- | --- | --- |
| P0 | Keep dense 1.5B as baseline pending falsification. | MoE is not default until infra and serving prove out. | `configs/pre2_dense_1_5b.yaml` remains the mainline contract. |
| P0 | Use lower-cost canary/POC before main compute. | Compute-limited path is 110M canary, 250M POC, 400M proxy, then 1.5B. | 110M/250M configs parse and have validated parameter budgets. |
| P0 | Align label/loss convention across loader, model, and trainer. | Packed loader emits shifted labels and `loss_mask`; model must consume that directly. | No double-shift in pre-2 data path. |
| P1 | Implement TorchTitan-compatible dense module adapter. | Current `myllm_pre2.model` is a local smoke module only. | Forward/loss parity test passes between local smoke and TorchTitan adapter. |
| P1 | Add shape and parameter accounting tests for all dense configs. | Include tied embeddings, GQA, RMSNorm, SwiGLU, RoPE, qk norm. | Parameter estimate remains within documented tolerance. |
| P1 | Add activation checkpointing and compile strategy study. | Compare eager, compile, selective activation checkpointing. | 400M proxy identifies stable memory/perf setting. |
| P2 | Add long-context continuation plan. | Foundation length first; continuation after base stability. | Continuation config has data, eval, and rope scaling gates. |

## 3. Training Runtime

| Priority | Task | Notes | Exit criteria |
| --- | --- | --- | --- |
| P0 | Keep `scripts/pre2_train.py` explicitly labeled as smoke only. | It must not be mistaken for the production trainer. | CLI help and docs say single-process smoke. |
| P1 | Create `src/myllm_pre2/torchtitan/` adapter. | Hold TorchTitan-specific AppState, config conversion, model factory, loss, and scheduler wiring. | 1 GPU synthetic run passes with TorchTitan primitives. |
| P1 | Add packed-corpus TorchTitan dataloader. | Preserve sequence id, token counts, segment ids, source ids, tokenizer hash, and loss mask. | Exact resume by data cursor passes under 1 GPU and 8 GPU. |
| P1 | Add single-process packed-corpus smoke before TorchTitan. | This is now present in `scripts/pre2_train.py` and should stay as a fast regression guard. | Tiny real packed-corpus fixture trains through shifted labels and loss mask. |
| P1 | Implement BF16 training path. | Smoke autocast is present; production BF16 still belongs in TorchTitan. | BF16 1 GPU TorchTitan smoke passes with finite gradients. |
| P1 | Implement WSD scheduler. | Use warmup-stable-decay with config-driven step math. | Scheduler state saves and restores exactly. |
| P1 | Add gradient clipping and global norm telemetry. | Basic smoke telemetry is present; production trainer still needs structured step logs. | Train logs include loss, z-loss, grad norm, tokens/sec, LR, memory. |
| P2 | Run stack bakeoff. | TorchTitan versus Megatron-Core versus DeepSpeed on same 100M-400M proxy and same packed data. | Pick production trainer with throughput, resume, and operational evidence. |

## 4. Checkpointing And Resume

| Priority | Task | Notes | Exit criteria |
| --- | --- | --- | --- |
| P0 | Keep current checkpoint as single-process smoke contract. | It is not DCP. | Docs and code comments state limitation. |
| P1 | Implement Torch Distributed Checkpointing. | Use AppState with model, optimizer, scheduler, train state, RNG, and data cursor. | Save/load round trip passes on 1 GPU. |
| P1 | Add exact same-topology resume test. | Single-process exact resume is present; production same-topology resume remains a DCP/TorchTitan gate. | Deterministic test passes for 1 GPU and fixed seed. |
| P2 | Add multi-rank DCP test. | World-size changes can be a later feature; same topology comes first. | 8 GPU interrupt/resume drill completes. |
| P2 | Add checkpoint manifest integrity. | Include config digest, tokenizer hash, data manifest hash, code commit, container digest. | Corrupted or mismatched artifacts fail closed. |

## 5. Data Corpus And Provenance

| Priority | Task | Notes | Exit criteria |
| --- | --- | --- | --- |
| P0 | Convert planning mixes into a source registry. | Initial machine-checkable registry exists; source approvals remain pending. | Each source has license, revision, URL/path, ingestion script, filters, token estimate. |
| P0 | Add license manifest and SPDX identifiers. | Include generated code/math, translations, teacher traces, and web sources. | Build refuses sources without license status. |
| P0 | Add synthetic-data accounting. | Synthetic cap must track translations, teacher-generated text, reasoning traces, and generated code/math. | Reports show synthetic share globally and by bucket. |
| P1 | Implement global dedup strategy. | Reuse proven Dolma/BFF or DCLM-style dedup; avoid unsupported in-memory MinHash at 1T scale. | Dedup report covers exact, near-duplicate, and benchmark decontam. |
| P1 | Add document quality filters. | Language ID, perplexity/quality classifier, boilerplate, toxicity, PII, code license. | Filter metrics saved per source revision. |
| P1 | Add packed-corpus metadata index. | Need source id, document id hash, spans, segment ids, tokenizer hash, and manifest digest. | Dataloader can prove every token's source bucket. |
| P2 | Build 100B-token pilot corpus. | Representative but small enough for repeated proxy runs. | Pilot corpus has manifest, dedup, decontam, and eval leakage report. |
| P2 | Build 1T+ training corpus. | Only after source registry and dedup are proven. | Data card renders from actual manifests, not static prose. |

## 6. Tokenizer

| Priority | Task | Notes | Exit criteria |
| --- | --- | --- | --- |
| P0 | Treat 131k SentencePiece-Unigram as provisional. | It may be right, but must be scored against alternatives. | Docs state provisional status. |
| P1 | Train tokenizer candidates: 64k, 96k, 131k; Unigram and BPE. | Use same data sample and same normalization policy. | Candidate artifacts and training scripts are reproducible. |
| P1 | Add tokenizer scorecard. | Bits/byte, fertility, Indic language coverage, code/math symbols, OOV/byte fallback rate, throughput. | Decision is data-backed, not preference-backed. |
| P1 | Freeze tokenizer before serious distillation. | Same-tokenizer KD requires stable tokenizer. | Tokenizer hash is embedded in corpus and checkpoint manifests. |
| P2 | Add tokenizer regression tests. | Verify normalization, special tokens, byte fallback, Indic scripts, code snippets. | Tests fail on accidental tokenizer drift. |

## 7. Evaluation And Release Gates

| Priority | Task | Notes | Exit criteria |
| --- | --- | --- | --- |
| P0 | Remove mock/non-empty scoring from release gate claims. | Existing scorecard path is not enough for release. | Real checkpoint path either runs eval or is clearly marked unimplemented. |
| P1 | Add `myllm_pre2.eval` prediction bridge. | Tiny checkpoint next-token eval is present; full lm-eval/LightEval wiring is still pending. | One local tiny checkpoint can run a toy eval end to end. |
| P1 | Define base eval suite. | MMLU-Pro/ProX, MILU, Belebele, GSM8K/MGSM, MATH, HumanEval+/MBPP+, BBH, IFEval, perplexity by domain. | Eval config lives in repo and is versioned. |
| P1 | Add contamination audit. | Benchmarks must be indexed before corpus pack. | Eval report includes contamination rate and blocked docs. |
| P2 | Add proxy-to-base transfer tracking. | Use 100M, 400M, 1.5B checkpoints. | Scaling curves and ablations are logged. |
| P2 | Add safety and refusal eval. | Required before public serving. | Release gate includes safety card and red-team results. |

## 8. Precision, Quantization, And Numerics

| Priority | Task | Notes | Exit criteria |
| --- | --- | --- | --- |
| P0 | Train canonical base in BF16 first. | INT4/INT8 are export/inference formats, not the first base-training precision. | Docs and configs do not imply 4-bit base training. |
| P1 | Add BF16/autocast path to smoke trainer. | Present for CPU/GPU smoke through `--precision bf16` and `--precision config`. | GPU smoke validates BF16 loss/backward. |
| P1 | Add FP8 shadow ablation. | Only after BF16 path is stable. Use torchao/Transformer Engine depending on selected runtime. | 1k-10k proxy steps match BF16 loss trend within tolerance. |
| P2 | Add PTQ exports: INT8 and INT4. | Use calibration set and eval parity. | Quantized model meets threshold versus BF16 checkpoint. |
| P2 | Add QAT only if PTQ fails. | QAT is expensive; do not schedule early without evidence. | Decision is backed by eval deltas. |

## 9. MoE Research Branch

| Priority | Task | Notes | Exit criteria |
| --- | --- | --- | --- |
| P0 | Do not sell 1B-total MoE as a win. | Research target should be 1B-active with larger resident params if infra allows. | MoE config stays research-only. |
| P1 | Build 100M-active MoE POC. | Test routing loss, capacity factor, expert parallelism, grouped GEMM, checkpoint, and serving implications. | POC trains and resumes on a small GPU setup. |
| P1 | Compare against dense proxy. | Same tokens, same tokenizer, same evals. | MoE continues only if quality/perf tradeoff is positive. |
| P2 | Evaluate Megatron-Core/NeMo for MoE. | TorchTitan may not be the right MoE runtime. | MoE runtime decision has throughput and operational proof. |

## 10. Serving And Export

| Priority | Task | Notes | Exit criteria |
| --- | --- | --- | --- |
| P1 | Add BF16 SafeTensors export. | First serving artifact should be lossless relative to training checkpoint. | Export/import parity test passes. |
| P1 | Add Hugging Face compatibility. | Needed for eval and serving ecosystem. | `AutoModel`/tokenizer load path works or a documented custom class exists. |
| P2 | Add vLLM serving path. | Primary inference backend candidate. | Local smoke generates text from exported checkpoint. |
| P2 | Add TensorRT-LLM and SGLang evaluation. | Compare throughput, quant support, and deployment complexity. | Serving ADR selects primary and fallback. |
| P2 | Add GGUF path for local quantized distribution. | Useful for 4-bit local tests if license permits. | GGUF export passes basic generation and eval subset. |
| P3 | Add ONNX only if downstream needs it. | Avoid work without a serving use case. | Tracked as optional. |

## 11. Governance, Legal, And Security

| Priority | Task | Notes | Exit criteria |
| --- | --- | --- | --- |
| P0 | Generate cards from artifacts. | Model/data cards should not be static placeholders. | Governance render script reads real manifests, evals, and config digests. |
| P0 | Add source license fail-closed behavior. | Unknown license should not enter training data. | Build fails before pack. |
| P1 | Map controls to NIST AI RMF, ISO/IEC 42001, EU AI Act GPAI, and SPDX 3.0.1. | Keep this operational, not just a document. | Evidence files exist for each required control. |
| P1 | Add security scan routine. | Include dependency scan, secret scan, artifact integrity, and supply-chain metadata. | Release checklist includes scan IDs and outcomes. |
| P2 | Add model risk register. | Misuse, bias, hallucination, data leakage, license, and privacy risks. | Release requires accepted residual risk. |

## 12. CI, Testing, And Quality

| Priority | Task | Notes | Exit criteria |
| --- | --- | --- | --- |
| P0 | Keep pre-2 tests focused and fast. | Config, guards, model, loader, checkpoint, smoke. | Local pre-2 test subset runs in seconds on CPU. |
| P0 | Add regression for loss-mask behavior. | Prevent doc-boundary tokens from contributing to loss. | Unit test fails if mask is ignored. |
| P1 | Add integration tests for packed data plus training step. | Use tiny real packed corpus fixture. | One optimization step consumes `PackedTorchBatch` correctly. |
| P1 | Add static checks. | Ruff/mypy can be staged gradually. | New pre-2 code passes selected lint gates. |
| P2 | Add GPU CI/manual gauntlet. | 1 GPU and 8 GPU jobs may be manual due cost. | Scripts emit machine-readable results. |

## 13. Hardware, Throughput, And Budget

| Priority | Task | Notes | Exit criteria |
| --- | --- | --- | --- |
| P0 | Keep hardware estimates marked as planning ranges. | Real throughput must come from proxy runs. | Docs do not claim measured throughput before measurement. |
| P1 | Run 8 GPU POC. | Validate stack, dataloader, checkpoint, BF16. | 8 GPU POC completes interrupt/resume drill. |
| P1 | Benchmark 16x H100/H200 plan. | Main candidate if calendar matters. | Tokens/sec and cost/token recorded. |
| P2 | Benchmark 32 GPU scaling. | Only if 16 GPU cannot hit timeline. | Scaling efficiency justifies added complexity. |
| P2 | Evaluate B200 only with price/perf proof. | Do not assume newest GPU is best operational choice. | Decision uses measured throughput and availability. |

## 14. Documentation And Operator Runbooks

| Priority | Task | Notes | Exit criteria |
| --- | --- | --- | --- |
| P0 | Keep README short and status-honest. | Link to detailed docs, avoid stale claims. | New contributor can run CPU tests. |
| P1 | Add pre-2 operator runbook. | Environment setup, corpus build, train launch, resume, eval, export. | Dry run by a second operator succeeds. |
| P1 | Add incident runbooks. | Non-finite loss, NCCL failure, checkpoint corruption, data mismatch, eval regression. | Each incident has triage steps and commands. |
| P2 | Add experiment report template. | Proxy studies should be comparable. | Every run records config, data, commit, hardware, evals, and conclusion. |

## Definition Of Pre-2 Foundation Ready

Pre-2 is not foundation-ready until all of these are true:

1. Dense baseline config, tokenizer, data mix, trainer, checkpoint, and eval stack are locked by ADR.
2. Source registry, license manifest, synthetic accounting, dedup, and decontam reports exist for the actual corpus.
3. BF16 proxy training runs are stable and resumable.
4. 8 GPU POC passes exact same-topology resume.
5. Release eval bridge runs real checkpoints, not mock scoring.
6. Hardware throughput is measured on the selected cluster type.
7. Governance cards are generated from real artifacts.
8. Serving/export path is proven at least for BF16 SafeTensors and one inference backend.
