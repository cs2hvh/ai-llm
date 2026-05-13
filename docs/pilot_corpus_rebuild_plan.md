# Stage 1 Pilot — Corpus Rebuild Plan (2026-05-13)

**Status**: drafted, awaiting execution on next CPU pod session.
**Target**: ~5B-token pretrain corpus so a 30B-token pilot is 6 epochs (acceptable repetition).
**Wall estimate**: ~6 hours on a 64-core CPU pod with `--max-parallel 8`.
**Cost estimate**: $6-18 depending on pod hourly rate.

---

## Why ~5B (and not the v1 base target of 1T)

Stage 1 is the 250M-parameter pilot whose job is to validate the training
stack end-to-end (FSDP correctness, watchdog, eval hook, R2 checkpoints,
data resume). Token budget per `configs/pilot_250m.yaml`:

| Target | Tokens |
|---|---|
| Smoke | 30B |
| Baseline | 50B |
| Stretch | 100B |

A 5B-token corpus pulled through the pilot at 30B target = 6 epochs.
Modern literature (Hoffmann et al., Muennighoff et al. "Scaling
Data-Constrained Language Models") shows 4-8 epochs is approximately as
useful as 1 epoch on a 4-8× larger corpus. Above ~10 epochs, returns
diminish sharply. 6 epochs is the sweet spot for a pilot.

The 1T target for Stage 3 base v1 is a separate (future) build with much
larger per-source budgets — explicitly out of scope per "don't focus on
v2 before v1 release" direction.

## Per-source budget table (target_tokens_per_source × share)

`scripts/run_parallel_builds.py --target-tokens-per-source 5_000_000_000`
will allocate as `5B × source.share`. Existing R2 sizes shown for context;
the rebuild writes to a NEW path (`corpus_v1_pilot/sources/`) so we don't
clobber the existing 1B-target build.

| Source | Share | Existing | Pilot target (5B × share) | Notes |
|---|---|---|---|---|
| **HuggingFaceFW/fineweb-edu** | 44.0% | 440M | **2.20B** | high quality web, dominant |
| **codeparrot/github-code-clean** | 18.0% | 180M | **900M** | swapped in for gated `bigcode/starcoderdata` |
| open_web_math/open-web-math | 7.0% | 70M | 350M | math |
| wikimedia/wikipedia (en) | 6.0% | 60M | 300M | encyclopedic |
| allenai/peS2o | 6.0% | 60M | 300M | academic |
| **pg19** | 5.0% | 40M | **250M** | books — may cap at ~250M (dataset is finite) |
| ai4bharat/sangraha (hin) | 4.0% | 40M | 200M | Hindi sovereign hedge |
| HuggingFaceH4/stack-exchange-preferences | 2.0% | 20M | 100M | Q&A — `text_field=question` still wastes the answer; tracked separately |
| mc4 (de) | 2.0% | 20M | 100M | multilingual |
| mc4 (es) | 1.5% | 15M | 75M | multilingual |
| mc4 (zh) | 1.5% | 15M | 75M | multilingual |
| mc4 (ar) | 1.5% | 15M | 75M | multilingual |
| mc4 (fr) | 1.5% | 15M | 75M | multilingual |
| **TOTAL** | **100.0%** | ~990M | **~5.0B** | |

## Critical change vs. existing `pretrain_mix.yaml`

`bigcode/starcoderdata` is GATED on HF (requires per-account access
acceptance + token authorization). For Stage 1 we substitute
`codeparrot/github-code-clean` (public, content-inline). The
substitution is captured in a new yaml `configs/data/pretrain_mix_pilot.yaml`
that the rebuild + compose flows reference. The original
`pretrain_mix.yaml` stays unmodified — when starcoderdata's gate is
eventually approved we can switch back without losing the existing
build.

## Execution (next CPU pod session)

```bash
# 1. Bring up a CPU pod (uses pod_launch_cpu.sh which is already
#    hardened against the install bugs).
bash scripts/pod_setup_apt.sh
# export R2 + HF + AWS_DEFAULT_REGION=auto then:

# 2. Set the build env vars:
export MYLLM_CORPUS_NAME=corpus_v1_pilot           # writes to corpus_v1_pilot/sources/
export MYLLM_DELETE_LOCAL=true                     # stream to R2, no disk fill
export MYLLM_SEQUENCE_LENGTH=4097                  # pilot is ctx=4096 + 1

# 3. Run the parallel builds. This drives 13 subprocesses (one per
#    source) under run_parallel_builds.py's coordination, with
#    --target-tokens-per-source 5B so each source's --target-tokens
#    is set to 5B * share. --max-parallel 8 keeps RAM in check.
source .venv/bin/activate
python scripts/run_parallel_builds.py \
    --pretrain-mix-config configs/data/pretrain_mix_pilot.yaml \
    --tokenizer-path artifacts/tokenizer_v1.json \
    --output-root /workspace/corpus/sources \
    --r2-prefix corpus_v1_pilot/sources \
    --target-tokens-per-source 5000000000 \
    --max-parallel 8 \
    --delete-local-after-upload \
    --production

# 4. After all 13 sources land in R2, compose into a single training-time
#    corpus. The compose pass interleaves sequences proportional to
#    target shares (deficit-driven mixing).
python scripts/compose_mixed_corpus.py \
    --sources-root /workspace/corpus/sources \
    --output-dir /workspace/corpus/pilot_train \
    --pretrain-mix-config configs/data/pretrain_mix_pilot.yaml \
    --sequences-per-shard 65536 \
    --strict-sources \
    --corpus-name corpus_v1_pilot_train
# Then upload to R2:
aws --endpoint-url "$S3_ENDPOINT_URL" s3 sync \
    /workspace/corpus/pilot_train s3://$S3_BUCKET/corpus_v1_pilot/train/
```

## Wall-time estimate

Codeparrot was 28 min for 180M tokens single-threaded on the dev box =
~107K tokens/sec. With `--max-parallel 8`:

| Source | Tokens | Single-thread wall |
|---|---|---|
| fineweb-edu | 2.20B | ~5.7 hr ← longest, dominates |
| codeparrot | 900M | ~2.3 hr |
| open_web_math | 350M | ~55 min |
| wikipedia / peS2o | 300M | ~47 min |
| pg19 | 250M | ~40 min (may cap earlier) |
| sangraha | 200M | ~30 min |
| stack_exchange / mc4_de | 100M | ~16 min |
| mc4_{es,zh,ar,fr} | 75M | ~12 min |

With 8 concurrent slots, the longest source (fineweb-edu) bottlenecks
wall time. **Expected total: ~6 hours.**

## Risk register for the rebuild

| Risk | Mitigation |
|---|---|
| pg19 caps out below 250M target | Acceptable. Build emits `target_tokens_unreachable` warning; we under-fill that slice. Total under-fill < 0.5% of corpus → negligible mix drift. |
| HF rate limiting on a single source | `run_parallel_builds.py` retries with backoff. If a source fails outright, we can re-run JUST that source via the regular `build_packed_corpus.py` CLI. |
| Decontam re-fires expensively | Each per-source build loads the 8+13gram indexes once at process start (~5 sec). Negligible. |
| Dedup MinHash false-flags new docs against rebuild | The MinHash dedupe is per-source state, NOT global. Old corpus_v1 docs aren't in the new build's MinHash store, so no false negatives. |
| R2 bandwidth saturation | We're shipping ~20 GB total to R2. RunPod typically gives ~100 MB/s effective → 3-4 min upload. Not a bottleneck. |
| pg19 / sangraha license verification | Already verified in earlier audit (`docs/governance/license_register.md`). No change. |

## What this unblocks

- Stage 1 pilot launch: `scripts/run_pretrain.py --model-config configs/pilot_250m.yaml --data-config configs/data/pretrain_mix_pilot.yaml --packed-corpus-root <r2-path-to-pilot-train> --total-steps <30B-tokens / step-batch> --eval-every 5000 ...`
- A real corpus for the eval hook's held-out batches (instead of synthetic).

## Out of scope (deferred to Stage 2/3 prep)

- Compose v2 1B corpus (the 1T+ release target). That's a separate build with much larger per-source budgets, not blocking pilot.
- StackExchange `text_field=question` fix (loses the answer content). Tracked as a separate rebuild item.
- Real-text teacher audit corpus. Stage 3 distillation prep.
- nemotron-CC addition (pending NVIDIA license approval). Would replace some of the FineWeb-Edu share.
