# MyLLM

Research and training stack for a sovereign small foundation model.

Current repository posture:

- `pre-1`: existing Keras/JAX pilot and validation stack.
- `pre-2`: planned production-grade redesign around a dense 1.5B-class model trained on 1.5T+ tokens.

## Current Plans

- [Project overview](docs/PROJECT_OVERVIEW.md)
- [Pre-2 final architecture and training plan](docs/PRE2_FINAL_PLAN_2026-05-14.md)
- [Pre-2 architecture decision record](docs/PRE2_ARCHITECTURE_DECISION.md)
- [Pre-2 stack migration plan](docs/PRE2_STACK_MIGRATION_PLAN.md)
- [Pre-2 module TODO backlog](docs/PRE2_MODULE_TODO_2026-05-14.md)
- [Pre-2 release and launch plan](docs/PRE2_RELEASE_PLAN.md)
- [Stage 3 data-prep Rust migration plan](docs/stage3_rust_migration_plan.md)
- [Governance docs](docs/governance/README.md)
- [Safety policy](docs/safety_policy.md)

## Current Pre-2 Decision

- Mainline architecture: dense decoder-only Transformer, approximately 1.49B parameters.
- Training target: 1.5T tokens, with 1.0T internal minimum and 3.0T stretch.
- Training stack: PyTorch 2.12+, TorchTitan, FSDP2, DTensor, Distributed Checkpointing.
- Data: staged portfolio built from high-quality educational web, curated web, multilingual/Indic, code, math/STEM, books/reference, Q&A/documentation, and bounded tagged synthetic data.
- Precision: train canonical base in BF16 first; evaluate FP8 training later; release BF16 plus quantized FP8/INT8/4-bit variants after calibration and eval.
- Distillation: heterogeneous-tokenizer top-K logit distillation is disabled. Use teacher-generated text/reasoning traces or same-tokenizer logit KD only.
- MoE: not the default. Treat 1B-active / 6B-8B-total MoE as a separate research branch.

Pre-2 config artifacts are present as planning contracts:

- `configs/pre2_dense_1_5b.yaml`
- `configs/pre2_dense_canary_110m.yaml`
- `configs/pre2_dense_poc_250m.yaml`
- `configs/pre2_dense_proxy_400m.yaml`
- `configs/pre2_moe_1b_active_research.yaml`
- `configs/data/pre2_mix_stage1.yaml`
- `configs/data/pre2_mix_stage2.yaml`
- `configs/data/pre2_mix_anneal.yaml`

These pre-2 configs are not consumed by the existing pre-1 JAX/Keras entrypoints.

## Repository Layout

```text
configs/          pre-1 runnable YAML plus pre-2 planning contracts
docs/             current planning, governance, safety, and migration docs
scripts/          training, corpus build, eval, audit, and utility CLIs
src/myllm/        tokenizer, data, model, training, eval, and orchestration code
tests/            unit and integration tests
```

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev,pre2]"
python scripts/smoke_test_env.py
python -m pytest
```

For Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1` and run the same `python -m pip` commands.

`requirements.txt` mirrors the pre-1 CPU/orchestration dependencies. `requirements-gpu.txt` is the existing pre-1 GPU pod profile for the JAX/Keras stack plus teacher-audit PyTorch. It is not the final pre-2 TorchTitan/PyTorch 2.12+ training image lock; that lock is a P0/P1 pre-2 deliverable.

After installing dependencies, validate the pre-2 planning configs with:

```bash
python scripts/pre2_config_check.py
python scripts/pre2_config_check.py --include-poc-ladder
python scripts/pre2_source_registry_check.py
```

Run the isolated pre-2 PyTorch smoke without touching the old JAX trainer:

```bash
python scripts/pre2_train.py --steps 1 --device cpu
python scripts/pre2_train.py --steps 1 --device cpu --precision config
python scripts/pre2_train.py --steps 1 --device cpu --checkpoint-dir artifacts/pre2-smoke-ckpt
python scripts/pre2_train.py --steps 1 --device cpu --resume-from-checkpoint artifacts/pre2-smoke-ckpt
python scripts/pre2_eval_toy.py --checkpoint-dir artifacts/pre2-smoke-ckpt --device cpu
```
