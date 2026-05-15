# MyLLM Pilot 250M v1

**Status**: ✅ COMPLETE (2026-05-15)
**Final eval**: val_loss **2.730**, val_ppl **15.34** (32 held-out batches, 32,768 tokens)
**Architecture**: 239M-param Llama-style decoder-only transformer
**Tokens trained on**: ~5.6 B (Stage 1: 4.98 B + Stage 1.5 decay re-iteration: ~640 M)
**Final checkpoint**: [`s3://llm-data/checkpoints/pilot-250m-v1-decay/step-000171990/`](./artifacts/manifest_step_171990.json)

This folder is the **time capsule** for Stage 1 of the MyLLM project — everything needed to understand what was built, reproduce the run, or load the final model. Shared scripts (`run_pretrain.py`, `eval_checkpoint.py`, `generate.py`) live in `scripts/` at the repo root; this folder collects the pilot-specific configs, results, and narrative.

## Quick read order

| If you want to | Read |
|---|---|
| Know what the pilot is + the headline numbers | This file (5 min) |
| See concrete results + sample generations | [`RESULTS.md`](./RESULTS.md) (10 min) |
| Reproduce any stage (corpus build → train → decay → eval → generate) | [`COMMANDS.md`](./COMMANDS.md) (15 min) |
| Understand the day-by-day chronology | [`TIMELINE.md`](./TIMELINE.md) (10 min) |
| Find the R2 paths for any artifact | [`R2_PATHS.md`](./R2_PATHS.md) (3 min) |
| Read the model card or governance docs | [`docs/`](./docs/) |

## What this pilot proved

The MyLLM project's thesis: *"a solo lead can ship a credible v1 small foundation model with off-the-shelf 2024-2025 research recipes."* The 250M pilot was Stage 1 — its job was to **validate the recipe**, not produce a deployable model.

Specific things validated end-to-end:
- **Architecture**: Llama-style decoder + GQA 3:1 + RMSNorm + QK-Norm + SwiGLU + RoPE + muP HP-transfer (base_width=256)
- **Training stack**: AdamW + WSD schedule (warmup → stable → decay) + Optax muP `multi_transform` + bf16 with z-loss
- **Safety / stability**: atomic NaN-revert (288 events handled cleanly), loss-spike watchdog (3σ soft, 6σ hard — never triggered hard)
- **Data pipeline**: 13-source corpus build (5.0 B tokens) → compose → packed-corpus on R2; dual-mode decontam (8+13 gram MinHash)
- **Sharding**: pure DP-replicated on 4×H200 SXM (no FSDP needed at 250M)
- **Checkpointing**: Orbax → R2 mirror every 5000 steps + atomic rollback on crash
- **Multilingual capability**: Hindi (sangraha source) confirmed working in generation tests

What did **not** ship (out of scope for Stage 1):
- Instruction tuning (SFT) — pilot is a BASE model
- Preference alignment (DPO / RLHF) — post-pretraining
- FSDP correctness for 1B+ scale — that's Stage 2's job
- Long-context extension past 8K
- Multimodality / tool use

## Final results in one table

| Metric | Value | Notes |
|---|---|---|
| Stage 1 final val_loss | 2.878 | before WSD decay; corpus exhausted at step 151,990 |
| Stage 1.5 (decay) final val_loss | **2.730** | the official model card number |
| Stage 1.5 final val_ppl | **15.34** | model narrows next-token prediction from 131,072 options to ~15 |
| Train loss (smoothed) at end | ~2.05 | post-decay |
| NaN-skip events total | 288 | 1.9 / 1000 steps — handled by atomic revert; no `hard_spike` rollbacks |
| Stage 1 wall time | ~12 hr | (with mid-run resume after int32 fix) |
| Stage 1.5 wall time | ~2 hr 18 min | clean run, no incidents |
| Total compute cost | ~$380 | 4×H200 SXM ($14/hr) × ~27 hr |

## File layout

```
pilots/250m_v1/
├── README.md            ← this file
├── RESULTS.md           ← detailed results + sample generations
├── TIMELINE.md          ← day-by-day narrative 2026-05-10 → 2026-05-15
├── COMMANDS.md          ← reproducer runbook (corpus build, train, decay, eval, generate)
├── R2_PATHS.md          ← R2 inventory: every artifact's exact location + size
├── configs/             ← frozen copies of pilot configs
│   ├── pilot_250m.yaml             ← the model spec
│   ├── pilot_250m_decay.yaml       ← Stage 1.5 variant (warmup_steps: 0)
│   └── pretrain_mix_pilot.yaml     ← 13-source data mix
├── artifacts/           ← small JSONs (fetched from R2)
│   ├── eval-final.json             ← post-hoc eval on step-151,990
│   ├── eval-final-decay.json       ← post-decay eval on step-171,990 (THE official number)
│   ├── manifest_step_151990.json   ← Stage 1 final checkpoint manifest
│   ├── manifest_step_171990.json   ← Stage 1.5 final checkpoint manifest
│   └── corpus_manifest.json        ← composed corpus top-level manifest
└── docs/                ← pilot-specific docs
    ├── pilot_corpus_rebuild_plan.md     ← why we rebuilt at seq_len=8193
    └── STATUS_2026-05-13.md             ← friend-reviewer-facing status doc
```

## What's NOT in this folder

These shared files at the repo root are used for the pilot but also for Stage 2 / Stage 3 (don't fork them here):
- `scripts/run_pretrain.py` — the training entry point
- `scripts/eval_checkpoint.py` — post-hoc eval
- `scripts/generate.py` — autoregressive generation (added during pilot, will serve later runs)
- `scripts/compose_mixed_corpus.py` — corpus composer
- `src/myllm/` — the entire codebase
- `docs/PROJECT_OVERVIEW.md` — project canon, kept current across stages
- `docs/SESSION_HANDOFF_2026-05-14.md` — cross-session pointer with full context

Bulk artifacts on R2 (NOT copied here, see `R2_PATHS.md`):
- Final checkpoint (2.65 GB) at `s3://llm-data/checkpoints/pilot-250m-v1-decay/step-000171990/`
- All 42 checkpoints (~110 GB) under the two checkpoint paths
- Composed corpus (20 GB) at `s3://llm-data/corpus_v1_pilot/train/`
- Per-source corpora (20 GB) at `s3://llm-data/corpus_v1_pilot/sources/`
- Tokenizer (4.79 MB) at `s3://llm-data/tokenizer/myllm-spm-unigram-131k-v2.json`
- Decontam indexes (75 MB) at `s3://llm-data/decontamination/`

## How to reproduce or extend

If you want to:

- **Re-eval the model**: see `COMMANDS.md` § "Post-hoc eval"
- **Generate text with it**: see `COMMANDS.md` § "Generate text"
- **Build the same corpus from scratch**: see `COMMANDS.md` § "Corpus build"
- **Run a fresh 250M training**: see `COMMANDS.md` § "Full training reproducer"
- **Continue training (Stage 1.6, etc.)**: needs multi-epoch reader (Phase 1.1 work)
- **Fine-tune for chat (SFT)**: out of scope for v1; documented as future work in handoff §6

## Lineage / next stages

- **Stage 1 (this pilot)**: 250M @ 5.6 B tokens — DONE
- **Stage 2** (planned): 1B @ 10–30 B tokens, ~3-5 days, ~$700-2000. Needs multi-epoch reader first.
- **Stage 3** (the real one): 1B @ 600 B tokens with DeepSeek-V4-Pro + Olmo-3-32B distillation, ~30 days, ~$13K
- **Post-release**: SFT + DPO + safety pass — separate workstream

See the canonical roadmap in `docs/SESSION_HANDOFF_2026-05-14.md` §14.9.

## Contact

- Lead: harshit.hv@samatva.com
- Repo: github.com/cs2hvh/ai-llm
- W&B project: harshit-hvpals-ahurasense/myllm
- R2 bucket: `llm-data` (endpoint in repo `.env`)
- Friend reviewer: external, async via `docs/review/` packets
