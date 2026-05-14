# Pre-2 Stack Migration Plan

Date: 2026-05-14
Status: implementation backlog

This is the execution plan for moving from the pre-1 JAX/Keras research stack
to the pre-2 PyTorch/TorchTitan training stack described in
`PRE2_FINAL_PLAN_2026-05-14.md`.

## Non-Negotiables

- Keep pre-1 runnable as a reference and pilot stack.
- Do not mix production JAX training and production PyTorch training in one environment.
- Do not launch a long run until resume, checkpoint, eval, and data-order replay are proven.
- Do not let heterogeneous-tokenizer logit KD re-enter the baseline.
- Treat every pre-2 config as non-runnable until its matching code path exists.

## Target Containers

| Container | Purpose | Main deps |
|---|---|---|
| `container-reference` | Existing pre-1 JAX/Keras code and tests | current `requirements*.txt` |
| `container-data` | Corpus build, filters, tokenization, manifests | Python, Rust/PyO3, pyarrow, tokenizers |
| `container-train` | TorchTitan/FSDP2 training | PyTorch, TorchTitan, NCCL, CUDA |
| `container-eval` | Generation/eval/export | vLLM or SGLang, lm-eval/lighteval, safetensors |

## Phases

| Phase | Goal | Exit criteria |
|---|---|---|
| P0 freeze | Freeze architecture, data-stage configs, and acceptance gates | ADR and configs merged |
| P1 model adapter | Implement TorchTitan-compatible dense model config and forward/loss | 1 GPU synthetic CE step passes |
| P2 checkpoint harness | Add distributed checkpoint save/restore and process-count restore tests | synthetic resume is exact |
| P3 dataloader | Port packed corpus reader to PyTorch with exact sequence-id resume | restart repeats the same next batch |
| P4 8-GPU smoke | Run 1B-shape synthetic and tiny real-corpus jobs under FSDP2 | no NaNs, checkpoint resume works |
| P5 proxy studies | Train 100M-400M proxies over 10B-50B tokens | tokenizer/data/LR choices selected |
| P6 1.5B canary | Run 10B-50B tokens on the target model | throughput and loss gates pass |
| P7 main run | Run 1.0T-1.5T tokens | go/no-go gates met |
| P8 context continuation | Extend to 16k/32k | long-context eval improves without core regressions |
| P9 release packet | Final eval, data card, model card, scorecard | release decision is evidence-backed |

## Code Work Items

| Work item | New target path | Notes |
|---|---|---|
| Pre-2 config loader | `src/myllm_pre2/config.py` | Keep separate from pre-1 `ModelConfig` until stable. |
| TorchTitan model adapter | `src/myllm_pre2/model.py` | Dense baseline only. |
| Packed PyTorch dataloader | `src/myllm_pre2/data/packed_loader.py` | Must preserve document/segment metadata. |
| DCP wrapper | `src/myllm_pre2/checkpoint.py` | Save/restore model, optimizer, scheduler, data cursor, RNG. |
| Training entrypoint | `scripts/pre2_train.py` | Fail closed on pre-1 configs. |
| Eval predict bridge | `src/myllm_pre2/eval.py` | No mock scorecards for release gates. |
| SafeTensors export | `scripts/pre2_export_safetensors.py` | BF16 reference export first. |
| Quantization exports | `scripts/pre2_quantize.py` | INT8/INT4 only after BF16 eval. |

## Test Work Items

| Test | Requirement |
|---|---|
| Config parse | pre-2 configs parse and reject pre-1-only fields |
| Param math | dense 1.5B estimate stays within accepted range |
| Forward/loss | 1 GPU synthetic step finite |
| Dataloader resume | same sequence ids after restart |
| Checkpoint resume | loss and optimizer state continue exactly on synthetic data |
| Process-count restore | save on one process count, restore on another supported count |
| Tokenizer hash | mismatch fails before training |
| Eval scorecard | release scorecard cannot use mocked predictions |
| KD guard | hetero-tokenizer top-K KD path fails closed |

## First Implementation Slice

The first code slice is deliberately small:

1. [x] Add `src/myllm_pre2/config.py` with a schema for `configs/pre2_dense_1_5b.yaml`.
2. [x] Add a param-count test for the dense config.
3. [x] Add `scripts/pre2_config_check.py` to validate model/data config pairs.
4. [x] Do not import TorchTitan yet.

That gives us a stable contract before adding distributed training code.

Next implementation slice:

1. [x] Add a minimal `src/myllm_pre2/model.py` dense module that mirrors the config contract.
2. [x] Add a 1-GPU synthetic forward/loss smoke test.
3. [x] Add a fail-closed guard that rejects heterogeneous-tokenizer top-K KD in pre-2 runs.

Next implementation slice:

1. [x] Add `scripts/pre2_train.py` as a CPU/1-GPU synthetic smoke entrypoint.
2. [x] Add the first packed-corpus PyTorch dataloader skeleton.
3. [x] Add checkpoint-state dataclasses for model, optimizer, scheduler, data cursor, and RNG.

Next implementation slice:

1. Add a DCP compatibility wrapper around the same checkpoint payload fields.
2. [x] Add a real packed-corpus fixture test for the PyTorch loader.
3. Add tokenizer-hash propagation from config/data manifest into smoke checkpoints.
