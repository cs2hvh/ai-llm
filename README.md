# MyLLM — sovereign 1B decoder-only LLM

Build pipeline for a from-scratch 1B-parameter decoder-only transformer.
Master plan: [`/root/PLAN.md`](../PLAN.md).

## Quickstart (orchestration VM)

```bash
# Python 3.11 recommended. Create venv and install runtime deps.
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# Smoke-test the environment (no GPU needed):
python scripts/smoke_test_env.py
```

GPU pods on RunPod use a different lockfile — see `requirements-gpu.txt` and `docs/phase0_bootstrap.md`.

## Layout

```
src/myllm/
  tokenizer/      BPE training + validation
  data/           streaming loaders, filters, dedupe, mixing
  model/          Keras layers (RoPE, RMSNorm, GQA, SwiGLU, decoder)
  training/       JAX mesh setup, train loop, optimizer, ckpt
  eval/           HF export + lm-eval-harness wrapper
  post_train/     SFT / DPO / reasoning / safety / tools / RAG
  quantize/       GGUF export pipeline
  runpod_orch/    RunPod SDK orchestration (renamed to avoid shadowing the `runpod` PyPI pkg)
configs/          YAML configs per phase
scripts/          one-shot CLIs
tests/            unit + smoke tests
docs/             phase-by-phase runbooks
artifacts/        local artifacts (gitignored — push to object storage)
```

## Stack note

We use **Keras 3** for modeling and **JAX** as the training backend (set via
`KERAS_BACKEND=jax`). TensorFlow remains usable as a Keras backend for tooling
and inference. JAX is the only realistic path to FSDP-style sharding at 1B
without months of custom DTensor work — see PLAN.md §3.

## Status

Phase 0 (bootstrap). No training has run. No compute has been spent.
