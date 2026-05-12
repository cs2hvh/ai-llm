# Phase 2 — Pilot 250M Launch Checklist
**Status: BLOCKED ON B2** — Phase 1 tokenizer ✅ shipped (`tokenizer/myllm-spm-unigram-131k-v2.json` in R2). Phase 2 needs the offline packed corpus (B2) before pilot can launch. See [`reviewer_qa_2026-05-12.md`](reviewer_qa_2026-05-12.md) §4 for the locked B2 design.

Pilot 250M validates the **full training stack on real GPU** at small scale before committing to the 1T-token base run. Goal is *stack validation*, not a useful released model. If the loss curve is sensible, the watchdog/checkpointing pipeline survives, and the R2 mirror works, Phase 2 has succeeded — independent of final loss.

---

## Pre-flight (must all be ✓ before launch)

- [x] **Phase 1 production tokenizer shipped** (2026-05-11)
  - `artifacts/tokenizer_v1.json` present locally
  - R2: `tokenizer/myllm-spm-unigram-131k-v2.json`
  - SHA-256 verified
  - 8/8 yaml validation round-trip tests pass
- [x] **Pretrain data mixture locked** (2026-05-12)
  - `configs/data/pretrain_mix.yaml` (FineWeb-Edu 44%, the-stack-v2 18%, Wiki 6%, pg19 5%, peS2o 6%, open-web-math 7%, stack-exchange 2%, Sangraha Hindi 4%, mc4 multilingual 8%)
  - Decontamination index covers 11 v1-gate benchmarks
- [ ] **B2 offline packed-shard generation** (PENDING — biggest open piece)
  - uint32 token shards, 512M tokens/shard, `tokens.bin` + `seq_meta.arrow` + `doc_meta.parquet` + `manifest.json`
  - Sharded CPU worker fleet (Rust tokenizers `encode_batch()`); target 5-20M tok/sec aggregate
  - Shard manifest must include tokenizer SHA256 (so training reads tokenizer-stamped shards only)
  - Per-source provenance carried per packed sequence (B3 schema, baked into seq_meta)
- [ ] **W&B project + run name** set: project `myllm`, run `pilot-250m-v1-<date>`
- [ ] **R2 checkpoint path** decided: `s3://llm-data/checkpoints/pilot-250m-v1/`
- [ ] **Credentials rotated** (RunPod, HF, W&B, R2) — pasted-in-chat creds are still active in `.env`. Rotate before booking a multi-day pod.

## GPU pod decision

| Option | SKU | Wall time (50B tok) | Cost @ list | Status |
|---|---|---|---|---|
| **A (preferred)** | 1× **NVIDIA B200** (180GB) | ~110 hr | ~$550 | confirmed in catalog 2026-05-11 |
| B (fallback) | 8× **NVIDIA H100 80GB HBM3** SXM | ~25 hr | ~$560 | confirmed in catalog 2026-05-11 |
| C (smoke-only) | 1× H100 SXM, 1B tokens | ~4 hr | ~$11 | for stack-only validation if budget tightens |

Choose A unless community-cloud capacity blocks it. If A blocks, fall back to B (FSDP=8 path).

## Launch command (paste-ready, B200)

```bash
# 1. Boot the pod (from this control-plane server)
cd /root/llm-build && set -a && source .env && set +a
.venv/bin/python -c "
import sys; sys.path.insert(0,'src')
from myllm.runpod_orch import GPUSku, PodSpec
from myllm.runpod_orch.client import RunPodClient
from myllm.runpod_orch.lifecycle import PodLifecycle
from myllm.runpod_orch.cost import CostLedger

spec = PodSpec(
    name='myllm-pilot-250m-v1',
    gpu=GPUSku.B200,
    gpu_count=1,
    image='runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04',
    container_disk_gb=200,
    volume_gb=500,
    cloud_type='COMMUNITY',
)
ledger = CostLedger(path='artifacts/pilot_ledger.jsonl', ceiling_usd=800.0)
client = RunPodClient()
lc = PodLifecycle(client=client, ledger=ledger, ready_timeout_seconds=900.0)
# (just print; actual context-managed launch happens from run_pretrain.py orchestration)
print('PodSpec validated.')
"

# 2. On the pod itself (after SSH-in), bootstrap + pretrain
bash scripts/bootstrap_pod.sh
KERAS_BACKEND=jax .venv/bin/python scripts/run_pretrain.py \
    --model-config configs/pilot_250m.yaml \
    --data-config configs/data/pretrain_mix.yaml \
    --tokenizer-r2-key tokenizer/myllm-spm-unigram-131k-v2.json \
    --total-steps $(python -c "print(50_000_000_000 // 4_194_304)") \
    --checkpoint-dir /workspace/checkpoints \
    --checkpoint-r2-prefix checkpoints/pilot-250m-v1 \
    --wandb-project myllm \
    --wandb-run pilot-250m-v1-$(date -u +%Y%m%d)
```

50B tokens at 4,194,304 tokens/step = 11,920 steps. Checkpoints every 1000 steps (12 total), last-3 + every-5000 kept.

## Watchdog / safety

- Watchdog spike threshold + auto-rollback already wired ([src/myllm/training/watchdog.py](../src/myllm/training/watchdog.py)).
- Cost ceiling on the pod: $800 (B200 @ $4.99/hr × 160 hr) — generous buffer past the $550 baseline. Anything over signals a stuck or hung run; investigate before paying more.
- On any rollback: `lr_recovery_multiplier *= 0.5` and `recovery_skip_batches = 100`. Max 3 recoveries before manual intervention.

## Smoke before the real pilot (recommended)

Run the orchestration smoke against B200 first (cost <$5) to confirm:
- B200 availability on community cloud at booking time
- New tokenizer artifact pulls from R2
- W&B logs land
- Cost ledger updates correctly

```bash
.venv/bin/python scripts/runpod_smoke.py --sku B200 --ceiling-usd 5.0
```

## Success criteria for Phase 2

| Gate | Threshold |
|---|---|
| Training loss curve at step 1000 | reasonable downward slope, no NaNs |
| Tokens/sec per GPU (B200) | ≥100K tok/s (else investigate FSDP/sharding) |
| Checkpoints uploaded to R2 | step 1000, 2000, 3000 all present in `s3://llm-data/checkpoints/pilot-250m-v1/` |
| Watchdog triggers correctly | inject a synthetic spike if no natural one occurs; verify rollback |
| Compression ratio held | per-language tokens/byte within 5% of held-out smoke baseline |
| Final eval (held-out)  | perplexity sensible (compare against SmolLM2 250M perplexity in same domain) |

If ≥5/6 gates pass: Phase 3 (1B base) is unlocked. If <5 pass: triage failures, do *not* scale up.

## Open prep items (parallel work while Phase 1 cooks)

- [ ] Implement packed-shard writer that consumes the pretrain mix and emits R2-mirrored arrow files
- [ ] Verify R2 multipart upload behavior at >100MB (`upload_file` currently uses single PUT; OK for tokenizer at 10-30MB but checkpoints will need multipart)
- [ ] Decide whether to push the custom Docker image to a registry (currently using `bootstrap_pod.sh` + stock pytorch image)
- [ ] Confirm B200 community-cloud capacity 24h before launch; if scarce, fall back to 8× H100 SXM
- [ ] Test data: 1B-token smoke shards generated and held in R2 for the optional stack-only smoke (option C above)
