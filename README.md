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

# Run the unit + integration test suite (~7s, 370 tests):
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
  research/       multi-agent research library (verify_candidates,
                  multi_source_lookup, parallel_audit) — see
                  docs/multi_agent_research.md
configs/          YAML configs per phase (wind_tunnel, pilot_250m, base_1b,
                  decay_phase_distillation, data/pretrain_mix)
scripts/          one-shot CLIs (run_pretrain, wind_tunnel_sweep,
                  build_decontamination_index, render_governance_cards,
                  research_cli, ...)
briefs/           YAML briefs for the research CLI (example_teacher_verify)
tests/            unit + integration tests (370 passing)
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

## Current status (2026-05-12)

| Phase | Status |
|---|---|
| 0 — bootstrap, RunPod orchestration smoke | ✅ done |
| 1 — production tokenizer (131k SPM-Unigram) | ✅ shipped |
| 2 — wind-tunnel sweep (Proxy A 67M + Proxy B 300M) | 🟡 sweep terminated; muP transfer validation pending Proxy B |
| 3 — pilot 250M | ⏳ pending B2 (offline packed corpus) |
| 4 — base 1B "internal v1" at 1T tokens | ⏳ pending Phase 3 |
| 5 — post-training (SFT/DPO/safety) | ⏳ |
| 6 — serving + quantization (GGUF) | ⏳ |

### Recent work (latest first)

- **2026-05-12** — Multi-agent research library: `verify_candidates` + `multi_source_lookup` + `parallel_audit` workflows + CLI + 29 tests. Adapted from Anthropic's [multi-agent research system blog post](https://www.anthropic.com/engineering/multi-agent-research-system). See [`docs/multi_agent_research.md`](docs/multi_agent_research.md).
- **2026-05-12** — Reviewer Q&A locked B2 design (uint32 tokens, 512M-token shards, simple seek index, sharded CPU workers w/ Rust tokenizers). See [`docs/reviewer_qa_2026-05-12.md`](docs/reviewer_qa_2026-05-12.md).
- **2026-05-12** — Red tests for the 4 "full-scale-only bug" coverage gaps + quarantine graceful-degradation fix. See [`docs/full_scale_bug_coverage_2026-05-12.md`](docs/full_scale_bug_coverage_2026-05-12.md). 370 tests passing.
- **2026-05-12** — P2 governance: decontamination extended to 11 benchmarks; auto-render of model_card/data_card from live configs (`scripts/render_governance_cards.py`).
- **2026-05-12** — Phase B re-audit fixes: state-dict preservation in `train_step`, data_position advancement in stable phase, micro_batch resolver, WSM merge template.
- **2026-05-12** — Teacher plan v2 locked: DeepSeek-V4-Pro-Base (MIT) + Olmo-3-32B (Apache-2.0). Mistral + Qwen3.6 dropped after license/modality verification. See [`docs/teacher_distillation_strategy.md`](docs/teacher_distillation_strategy.md).
- **2026-05-12** — Phase B batch 1: Orbax template-aware restore (B1), decay-phase activation + α-annealing (B7/B8), quarantine writer (B6), governance scaffolding (B9).
- **2026-05-12** — Phase A: 6 P0 integration bugs fixed (atomic NaN-skip, segment_ids end-to-end, sequence-length resolver, alpha-from-batch, token-weighted mixture sampling, bf16 teacher-cache dtype fix).

See [`docs/project_handoff_2026-05-11.md`](docs/project_handoff_2026-05-11.md) for the full context dump (handoff brief).

## External reviews (audit trail)

- [`docs/MyLLM_Repo_Technical_Review_2026-05-12.docx`](docs/MyLLM_Repo_Technical_Review_2026-05-12.docx) — first colleague's code review
- [`docs/external_review_2026-05-12_enterprise.md`](docs/external_review_2026-05-12_enterprise.md) — enterprise strategy review
- [`docs/reviewer_qa_2026-05-12.md`](docs/reviewer_qa_2026-05-12.md) — follow-up Q&A locking B2 design choices

Per the "verify-before-locking" rule, every external claim that influenced a code or config change has been WebFetch-verified before the lock.
