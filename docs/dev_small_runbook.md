# dev-small mini-pilot runbook

**Branch**: `dev-small`
**Purpose**: end-to-end lifecycle validation. Run a 100M model for under 1B tokens, see what a COMPLETED training run looks like before the 250M pilot wraps (~22 hr from 2026-05-14 02:30 UTC).

**Not a production run.** Loss/perplexity numbers won't be directly comparable to the 250M pilot — different scale, different tokens-per-param ratio. The goal is to validate the FULL pipeline end-to-end:

- Model build + JIT
- Multi-GPU sharding (if applicable)
- Train loop + checkpoint cycle
- Eval hook firing across multiple eval points
- W&B run lifecycle (start → checkpoints → finish)
- Final checkpoint save + R2 mirror
- Subsequent release scorecard pipeline against the final checkpoint

If this run completes cleanly, we know the SAME post-training flow will work for the real 250M pilot tomorrow.

## Model

100M params (vs 250M pilot). Same architecture family, downscaled:

| Param | pilot_250m | pilot_100m |
|---|---|---|
| hidden_dim | 768 | **512** |
| layers | 16 | **10** |
| num_heads | 12 | **8** |
| num_kv_heads | 4 | **2** |
| head_dim | 64 | 64 |
| ffn_dim | 3072 | **2048** |
| context_length | 8192 | 8192 |
| muP width_mult | 3.0 | **2.0** |
| warmup_steps | 2000 | **500** |
| total params | ~249M | **~99M** |

Same tokenizer (131k), same corpus, same WSD schedule shape, same peak_lr (3e-4 — muP transfer handles the scale gap).

## Token budget

Target: **~500M tokens** (well under 1B). Specifics depend on pod:

| Pod | DP | mb_global | tokens/step | Steps for 500M | Wall (est) |
|---|---|---|---|---|---|
| 1× H100/H200 | 1 | 1 | 8,192 | 61,000 | ~2-3 hr |
| 4× H100/H200 | 4 | 4 | 32,768 | 15,259 | ~25 min |
| 4× B200 | 4 | 4 | 32,768 | 15,259 | ~15 min |

For the run command below I use `--total-steps 15000`. Adjust if you want longer/shorter.

## Setup on the new pod (same as 8×H200 procedure)

1. Set env vars (R2 + HF + WANDB + NCCL):

```bash
export AWS_ACCESS_KEY_ID='<your-r2-key>'
export AWS_SECRET_ACCESS_KEY='<your-r2-secret>'
export S3_BUCKET='llm-data'
export S3_ENDPOINT_URL='https://<your-account>.r2.cloudflarestorage.com'
export AWS_DEFAULT_REGION=auto
export HF_TOKEN='<your-hf-token>'
export WANDB_API_KEY='<your-wandb-key>'
export NCCL_NVLS_ENABLE=0
export NCCL_IB_DISABLE=1
unset LD_LIBRARY_PATH
```

2. Clone + checkout `dev-small`:

```bash
git clone -b dev-small https://github.com/cs2hvh/ai-llm.git /workspace/llm-build
cd /workspace/llm-build
```

3. Install (~10-15 min, or skip if pod template already has it):

```bash
bash scripts/pod_setup_apt.sh
bash scripts/pod_launch_gpu.sh
source /workspace/llm-build/.venv/bin/activate
```

4. Pull tokenizer + corpus from R2:

```bash
mkdir -p artifacts
aws --endpoint-url "$S3_ENDPOINT_URL" s3 cp \
    s3://$S3_BUCKET/tokenizer/myllm-spm-unigram-131k-v2.json \
    artifacts/tokenizer_v1.json

mkdir -p /workspace/corpus_pilot_train
AWS_MAX_ATTEMPTS=10 AWS_RETRY_MODE=adaptive aws --endpoint-url "$S3_ENDPOINT_URL" s3 sync \
    s3://$S3_BUCKET/corpus_v1_pilot/train/ \
    /workspace/corpus_pilot_train/
```

5. Verify corpus complete (10 shards, ~19 GB total, all shards with 4 files):

```bash
for d in /workspace/corpus_pilot_train/shard-*/; do
    echo "$(basename $d): $(ls $d | wc -l) files, $(du -sh $d | cut -f1)"
done
```

If any shard is incomplete (3 files, 17 MB), use the boto3 fix from earlier flows.

## Launch the mini pilot

In a tmux session (`tmux new -s dev-small`):

```bash
# Match --micro-batch-override to your DP count (must be divisible by DP):
#   1 GPU → mb=1
#   4 GPU → mb=4
#   8 GPU → mb=8

python scripts/run_pretrain.py \
    --model-config configs/pilot_100m.yaml \
    --data-config configs/data/pretrain_mix_pilot.yaml \
    --tokenizer-path artifacts/tokenizer_v1.json \
    --packed-corpus-root /workspace/corpus_pilot_train \
    --run-name dev-small-100m-v1-2026-05-14 \
    --total-steps 15000 \
    --micro-batch-override 4 \
    --log-every 50 \
    --eval-every 1000 \
    --eval-n-batches 16 \
    --checkpoint-every 1000 \
    --checkpoint-root /workspace/ckpt/dev-small-100m \
    --checkpoint-r2-prefix checkpoints/dev-small-100m \
    2>&1 | tee /workspace/dev-small.log
```

**eval-every=1000 / checkpoint-every=1000** mirrors the 250M's cadence relative to total length (250M run does eval every 5000 of 229000 = 2.2%; this run does every 1000 of 15000 = 6.7%, slightly more frequent so we see more eval points in the short run).

Detach tmux when stable: Ctrl-B then D.

## What success looks like

- `training_complete` event fires
- Final checkpoint exists at `/workspace/ckpt/dev-small-100m/step-000015000/`
- R2 has the final checkpoint at `s3://llm-data/checkpoints/dev-small-100m/step-000015000/`
- W&B run shows full curve, final eval, `Run finished` state
- Loss reached some value < 4.0 (rough — 100M @ 500M tokens won't get as low as 250M @ 30B, but should be in the 3.0-4.0 range)
- 0-5 NaN-skip events total (similar to the 250M pilot's rate at 5/1000)

## After it completes

Run the release scorecard against the final checkpoint (this is the OTHER thing we want to validate before the real pilot):

```bash
python scripts/build_release_scorecard.py \
    --checkpoint /workspace/ckpt/dev-small-100m/step-000015000 \
    --model-config configs/pilot_100m.yaml \
    --tokenizer-path artifacts/tokenizer_v1.json \
    --use-mock-predict \
    --benchmarks mmlu-pro,gsm8k \
    --sample-size 50 \
    --output-dir /workspace/scorecard-dev-small
```

(Use `--use-mock-predict` because real benchmark scoring needs the predict_fn implementation that's still pending. The mock just exercises the scorecard pipeline — JSON + Markdown output.)

If the scorecard outputs JSON + Markdown cleanly, the full post-training pipeline is validated.

## When to merge back to main

After dev-small finishes successfully, this branch is NOT meant to merge back — it's a one-off lifecycle test. Just keep the branch around for reference. The `configs/pilot_100m.yaml` can stay; it's a useful "quick lifecycle test" config to keep in the repo for future debugging.

If you want it back on main:

```bash
git checkout main
git merge dev-small  # or cherry-pick the config file specifically
```
