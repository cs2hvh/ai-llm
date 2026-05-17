# MyLLM — sovereign 1B decoder-only LLM

Build pipeline for a from-scratch 1B-parameter decoder-only transformer.
Master plan: [`/root/PLAN.md`](../PLAN.md).

## Quickstart (orchestration VM)

```bash
# Python 3.11+ recommended. Create venv and install runtime deps.
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# Smoke-test the environment (no GPU needed):
python scripts/smoke_test_env.py

# Run the unit + integration test suite (~7s):
python -m pytest
```

GPU pods on RunPod use a different lockfile — see `requirements-gpu.txt`. Pod bootstrap is automated via `scripts/bootstrap_pod.sh` (invoked by `pod_launch.sh` on first SSH).

## Layout

```
src/myllm/
  tokenizer/      SPM-Unigram training + validation
  data/           streaming loaders, filters, dedupe, mixing, packing,
                  decontamination, prompt loaders, teacher cache
  model/          Keras 3 layers (RoPE, RMSNorm, GQA, SwiGLU, decoder)
  training/       JAX mesh, train loop, optimizer (muP), Orbax checkpoint,
                  WSM merge, decay-phase distillation, quarantine writer
  eval/           lm-eval-harness wrapper + multilingual benchmark adapters
  post_train/     SFT / DPO / reasoning / safety / tools / RAG (Phase 4)
  quantize/       GGUF export (Phase 6)
  runpod_orch/    RunPod SDK orchestration
configs/          YAML configs per phase (wind_tunnel, pilot_250m, base_1b,
                  decay_phase_distillation, data/pretrain_mix)
scripts/          one-shot CLIs (run_pretrain, wind_tunnel_sweep,
                  build_decontamination_index, render_governance_cards, ...)
tests/            unit + integration tests (full pytest suite passing)
docs/             phase runbooks + governance + external reviews
docs/governance/  EU AI Act / ISO 42001 / NIST AI RMF / DPDP artifacts
artifacts/        local artifacts (gitignored — push to R2 instead)
```

## Stack

- **Modeling**: Keras 3 with **JAX backend** (`KERAS_BACKEND=jax`). JAX is the only realistic path to FSDP-style sharding at 1B without months of custom DTensor work — see PLAN.md §3.
- **Optimizer**: AdamW via Optax, muP per-parameter LR scaling through `optax.multi_transform`.
- **Checkpointing**: Orbax sharded, manifest-as-completion-marker for atomic R2 mirror.
- **Distillation**: top-K=8 cached teacher logits (bf16 packed in uint16) in Arrow shards; decay-phase activation at 0.85 × total_steps with α annealing 0.7 → 0.3.
- **Decontamination**: 13-gram xxhash64 index over 11 v1-gate benchmarks (MMLU-ProX/Pro, Belebele, MILU, HumanEval+, MBPP+, GSM8K, MATH, MGSM, BBH, IFEval).
- **Tokenizer**: SentencePiece Unigram, 131k vocab, NFKC + Metaspace + byte_fallback.

## Current status (2026-05-17)

| Phase | Status |
|---|---|
| 0 — bootstrap, RunPod orchestration smoke | ✅ done |
| 1 — production tokenizer (131k SPM-Unigram) | ✅ shipped |
| 2 — wind-tunnel sweep (Proxy A 67M + Proxy B 300M) | ✅ closed by **C3 μP/LR sweep at 1B-shape** (peak_lr=3e-4 wins monotonically across 3-LR sweep on 4×B200) — **muP transfer 250M → 1B CONFIRMED 2026-05-16** |
| 3 — pilot 250M | ✅ done. val_loss 2.7303 / val_ppl 15.34. Stage 1 + Stage 1.5 decay |
| 4 — base 1B Stage 2 rehearsal (10-30B tokens) | 🔄 ready, hardware/budget decision pending |
| 4 — base 1B Stage 3 base run (~600B-1T tokens) | ⏳ blocked on Stage 2 |
| 5 — post-training (SFT/DPO/safety) | ⏳ |
| 6 — serving + quantization (GGUF) | ⏳ |

### Where to read next

- [`docs/SESSION_HANDOFF.md`](docs/SESSION_HANDOFF.md) — live current state + plan ahead
- [`docs/DESIGN.md`](docs/DESIGN.md) — architecture + algorithms reference (with Mermaid diagrams)
- [`docs/review/POST_PILOT_REVIEW_2026-05-15.md`](docs/review/POST_PILOT_REVIEW_2026-05-15.md) — current reviewer packet
- [`docs/archive/`](docs/archive/) — historical reviews, pre-pilot plans, old handoffs

Per the "verify-before-locking" rule, every external claim that influenced a code or config change is WebFetch-verified before the lock.
