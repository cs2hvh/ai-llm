# Wind-Tunnel Sweep — Launch Checklist
**Status: SWEEP TERMINATED 2026-05-12 PM** (per user direction; pod stopped). Proxy B 300M transfer validation now the gate before pilot launch. Re-launch of Proxy A sweep pending.

The wind-tunnel sweep is the **last code-side gate before Phase 2 pilot 250M**. It runs a 10-cell grid (5 LRs × 2 init_stds) against a 67M proxy model (`configs/wind_tunnel.yaml`) to find the optimal (peak_lr, init_std) under muP. Per the muP transfer law, the optimum at the 67M proxy transfers zero-shot to pilot 250M (width_mult=4) and base 1B (width_mult=8).

**Reviewer addition (2026-05-12):** before trusting the transfer to the 1B base run, validate via Proxy B (300M, width_mult=4 — same width_mult as pilot) at the chosen (LR*, init*). One cell, 500M tokens, ~$11-20. See `configs/wind_tunnel_b.yaml`.

**Budget**: ~$30-50 for Proxy A (10 cells × 200M tokens each, ~$3-5/cell on H200 SXM) + ~$11-20 for Proxy B single cell. Wall time per cell ~1 hr on H200 SXM at $3.99/hr.

**SKU choice (2026-05-11)**: B200 is **Unavailable** on Secure Cloud per the RunPod dashboard. H200 SXM (141 GB HBM3) is **Low** availability at $3.99/hr — same throughput class as B200, ~33% cheaper, and currently grabbable. Use this. Verified live via `runpod.get_gpu("NVIDIA H200")`.

---

## Pre-flight (must all be ✓)

- [ ] **Production tokenizer present**: `artifacts/tokenizer_v1.json` exists locally OR `--tokenizer-key tokenizer/myllm-spm-unigram-131k-v2.json` is set (the script auto-downloads from R2)
- [ ] **Wind-tunnel config validated**: `python -c "from myllm.model import ModelConfig; ModelConfig.from_yaml('configs/wind_tunnel.yaml')"` succeeds
- [ ] **Dry-run works locally**: `python scripts/wind_tunnel_sweep.py` prints 10 cells with `--peak-lr-override` and `--init-std-override` flags
- [ ] **Tests green**: `pytest tests/test_wind_tunnel.py` returns 20/20 pass
- [ ] **R2 credentials**: `.env` has `R2_*` vars set; `aws s3 ls $S3_BUCKET/checkpoints/` returns 0
- [ ] **W&B disabled** in cell command (the sweep deliberately doesn't log — verified by `--no-wandb` in `cell_command()`)
- [ ] **Pretrain data config accessible**: `configs/data/pretrain_mix.yaml` parses, all HF datasets reachable from the pod (test on a small streaming pull first)

## Pod spec

Prices are live as of **2026-05-11** (verified via `runpod.get_gpu(...)`). Secure cloud unless noted — secure shows up as the default on the user's dashboard.

| Option | SKU | Per-cell wall | $/hr | Per-cell cost | Total | Notes |
|---|---|---|---|---|---|---|
| **A (preferred)** | 1× H200 SXM secure, sequential | ~1.1 hr | $3.99 | ~$4.40 | **~$44** | 141 GB HBM3. "Low" availability — book promptly. **1 max GPU/pod** at this SKU. |
| B (cheaper alt) | 1× H200 SXM community, sequential | ~1.1 hr | $3.59 | ~$4.00 | ~$40 | $0.40/hr cheaper; community pods can preempt — avoid for long-running cells |
| C (parallel) | 10× 1× H200 SXM, one per cell | ~1.1 hr | $3.99 | ~$4.40 | ~$44 | Same cost, ~1 hr end-to-end. Requires 10 simultaneous "Low"-availability pods — likely blocked |
| D (fallback) | 1× H100 SXM community, sequential | ~1.6 hr | $2.69 | ~$4.30 | ~$43 | If H200 stays unavailable. ~1.5× slower → ~16 hr end-to-end |
| ~~E (was preferred)~~ | ~~1× B200~~ | — | — | — | — | **B200 is "Unavailable" on Secure Cloud as of 2026-05-11.** |

Default to **A**. Only escalate to C if you need the result the same day AND can secure 10 pods.

## Launch sequence (Option A — single H200 SXM pod, sequential)

```bash
# 1. From control plane: boot a 1× H200 SXM secure pod with cost ceiling $55.
#    (~$4.40/cell × 10 cells = ~$44 expected; ceiling has $11 buffer.)
.venv/bin/python -c "
import sys; sys.path.insert(0,'src')
from myllm.runpod_orch import GPUSku, PodSpec
from myllm.runpod_orch.client import RunPodClient
from myllm.runpod_orch.lifecycle import PodLifecycle
from myllm.runpod_orch.cost import CostLedger

spec = PodSpec(
    name='myllm-wind-tunnel',
    gpu=GPUSku.H200_SXM,
    gpu_count=1,
    image='runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04',
    container_disk_gb=120,
    volume_gb=200,
    cloud_type='SECURE',     # H200 SXM 'Low' availability on secure cloud; B200 Unavailable
)
ledger = CostLedger(path='artifacts/wind_tunnel_ledger.jsonl', ceiling_usd=55.0)
client = RunPodClient()
lc = PodLifecycle(client=client, ledger=ledger, ready_timeout_seconds=900.0)
print('PodSpec validated. Launch via runpod_orch context manager next.')
"

# 2. SSH to the pod, then on the pod:
bash scripts/bootstrap_pod.sh

# Pull the tokenizer from R2 (or have run_pretrain do it via --tokenizer-key)
aws s3 cp s3://$S3_BUCKET/tokenizer/myllm-spm-unigram-131k-v2.json artifacts/tokenizer_v1.json

# 3. Execute the sweep (sequential, all 10 cells)
KERAS_BACKEND=jax .venv/bin/python scripts/wind_tunnel_sweep.py \
    --execute \
    --output artifacts/wind_tunnel_results.json \
    --tokenizer-path artifacts/tokenizer_v1.json \
    --artifact-root artifacts
```

The script writes the manifest after each cell completes, so a mid-sweep crash is recoverable — re-run with `--execute` and it will overwrite finished cells' status (TODO: add `--skip-done` flag if this becomes an issue in practice).

## In-flight monitoring

Per-cell logs land at `artifacts/wind_tunnel/<cell_id>/train.log`. Watch the most recent one:

```bash
tail -f artifacts/wind_tunnel/$(ls -t artifacts/wind_tunnel/ | head -1)/train.log | grep -i loss
```

**Red flags** (kill the run, debug before retrying):
- Loss is NaN/inf within the first 100 steps → either init_std too high or LR too high; bad cell but should not crash the sweep
- All cells finish with loss > 8.0 → tokenizer not loaded correctly, or data pipeline broken
- Cells take >2 hours → throughput regression; check that JAX is using the GPU (`jax.devices()` should show CUDA)

**Yellow flags** (note but don't abort):
- One or two cells diverge to high loss — expected at the extremes of the grid (8e-3 LR may be too hot)

## Post-run: select optimum + write HPs back

After all 10 cells finish:

```bash
# Re-parse logs (idempotent; safe to run multiple times).
.venv/bin/python scripts/wind_tunnel_sweep.py --collect \
    --output artifacts/wind_tunnel_results.json
```

This prints a table like:

```
cell_id            |    peak_lr |  init_std |     tokens | final_loss | elapsed | status
---------------------------------------------------------------------------------------
lr5e-04_init1e-02  |   5.00e-04 |  1.00e-02 |  200000000 |     3.8421 |   3850s | done
lr5e-04_init2e-02  |   5.00e-04 |  2.00e-02 |  200000000 |     3.8055 |   3851s | done
lr1e-03_init1e-02  |   1.00e-03 |  1.00e-02 |  200000000 |     3.6712 |   3870s | done
lr1e-03_init2e-02  |   1.00e-03 |  2.00e-02 |  200000000 |     3.6418 |   3872s | done
lr2e-03_init1e-02  |   2.00e-03 |  1.00e-02 |  200000000 |     3.5891 |   3850s | done
lr2e-03_init2e-02  |   2.00e-03 |  2.00e-02 |  200000000 |     3.5610 |   3855s | done   ← typically best
lr4e-03_init1e-02  |   4.00e-03 |  1.00e-02 |  200000000 |     3.6020 |   3863s | done
lr4e-03_init2e-02  |   4.00e-03 |  2.00e-02 |  200000000 |     3.5985 |   3858s | done
lr8e-03_init1e-02  |   8.00e-03 |  1.00e-02 |  200000000 |     4.1230 |   3866s | done
lr8e-03_init2e-02  |   8.00e-03 |  2.00e-02 |  200000000 |     4.0987 |   3851s | done

best: lr2e-03_init2e-02  peak_lr=2.00e-03  init_std=2.00e-02  loss=3.5610
```

Sanity check the curve shape:
- LR should trace a **U-curve**: loss decreases from 5e-4 → optimum, then increases at 8e-3. If it's monotone, the optimum is at an edge and you need to extend the grid.
- init_std rows should be **roughly parallel** (init effect is small); if 0.01 wins by >0.1 at every LR, the init dimension is mis-bracketed.

Then write the winning HPs into the pilot + base configs:

```bash
# Manual: edit configs/pilot_250m.yaml + configs/base_1b.yaml
#   lr_schedule.peak_lr        = <best.peak_lr>
#   init_std                   = <best.init_std>
#   mup.base_width             = 256   (matches wind_tunnel.yaml; required for HP transfer)
#
# Also add a comment with the wind-tunnel result reference, e.g.:
#   # peak_lr from wind-tunnel sweep 2026-05-13 (cell lr2e-03_init2e-02, loss=3.561)
```

(Future cleanup: a small `scripts/apply_wind_tunnel_hps.py` could do this mechanically. Not built yet — manual edit is fine for now since this is a one-time event per architecture change.)

## Archive results to R2

```bash
aws s3 cp artifacts/wind_tunnel_results.json \
    s3://$S3_BUCKET/wind_tunnel/results_$(date -u +%Y%m%d).json
aws s3 sync artifacts/wind_tunnel/ \
    s3://$S3_BUCKET/wind_tunnel/runs_$(date -u +%Y%m%d)/
```

## Failure modes

| Symptom | Probable cause | Fix |
|---|---|---|
| `KERAS_BACKEND=jax` not set, `keras.src` import errors | Bootstrap script missed env export | `export KERAS_BACKEND=jax` then re-run |
| `OutOfMemoryError` at step ~50 | Sequence length 2048 too long for H200 SXM with batch=8 (141GB should be plenty for a 30M model, so OOM here means a leak) | Drop `micro_batch_per_device` to 4 in pretrain_mix.yaml; tokens/step halves, total_steps doubles. If it still OOMs, check `jax.devices()[0].memory_stats()` for the leak |
| HF dataset fails to stream | Network blip on community pod | Re-run failed cell only with `--lr-grid <single> --init-grid <single>` |
| Loss = NaN at step 0 | init_std too large for hidden_dim=384 | Skip that cell in the manifest; muP literature says 0.02 is upper bound |
| Sweep manifest corrupted | mid-write crash | Delete `artifacts/wind_tunnel_results.json`, re-run from scratch |

## Done definition

- [ ] All 10 cells `status: done`
- [ ] `select_best()` returns a cell with finite loss
- [ ] HPs written back to `pilot_250m.yaml` and `base_1b.yaml` with provenance comments
- [ ] `wind_tunnel_results.json` mirrored to R2
- [ ] Phase 2 pilot launch checklist unblocked

---

## Why this matters (one-paragraph rationale)

Without the wind-tunnel sweep, pilot 250M and base 1B inherit a guessed peak_lr — most likely too low (Llama-3-conservative ~3e-4) which would waste ~30% of the training budget on understepping the loss. With muP + a 30M-proxy sweep, we pay $30-40 to find an LR that's measured-optimal at 30M and *provably transfers* to 1B per the muP transfer law. This is the cheapest possible insurance for the $80-115K Phase 3 base 1B run.
