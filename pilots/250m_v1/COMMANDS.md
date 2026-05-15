# MyLLM Pilot 250M v1 — Reproducer Runbook

This is the runbook to reproduce any stage of the pilot OR to extend it (e.g., load the final model for new work). All commands assume:

- A pod with the canonical pip stack (torch 2.7.1 + jax[cuda12] 0.4.38) at `/workspace/llm-build/`
- Env vars set: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `S3_BUCKET=llm-data`, `S3_ENDPOINT_URL`, `AWS_DEFAULT_REGION=auto`, `HF_TOKEN`, and (optional) `WANDB_API_KEY`
- `unset LD_LIBRARY_PATH; export NCCL_NVLS_ENABLE=0; export NCCL_IB_DISABLE=1` for RunPod H100/H200 multi-GPU

For pod bring-up from scratch, see `docs/SESSION_HANDOFF_2026-05-14.md` §11.

---

## Post-hoc eval (cheapest, ~$1, ~5 min on 1×H100)

Load the final checkpoint, compute val_loss on 32 held-out batches.

```bash
# Pull the final checkpoint from R2 (one-time, ~2.65 GB, ~30 sec)
mkdir -p /workspace/ckpt/pilot-250m-v1-decay
AWS_MAX_ATTEMPTS=10 AWS_RETRY_MODE=adaptive aws --endpoint-url "$S3_ENDPOINT_URL" s3 sync \
    s3://$S3_BUCKET/checkpoints/pilot-250m-v1-decay/step-000171990/ \
    /workspace/ckpt/pilot-250m-v1-decay/step-000171990/

# Pull tokenizer (one-time, ~5 MB)
mkdir -p artifacts
aws --endpoint-url "$S3_ENDPOINT_URL" s3 cp \
    s3://$S3_BUCKET/tokenizer/myllm-spm-unigram-131k-v2.json artifacts/tokenizer_v1.json

# Pull the corpus (only the first shard is needed for 32 batches, but easiest is full sync)
mkdir -p /workspace/corpus_pilot_train
aws --endpoint-url "$S3_ENDPOINT_URL" s3 sync \
    s3://$S3_BUCKET/corpus_v1_pilot/train/ /workspace/corpus_pilot_train/

# Run eval (uses scripts/eval_checkpoint.py at commit ca1c40b+ for G6 reshard support)
python scripts/eval_checkpoint.py \
    --checkpoint /workspace/ckpt/pilot-250m-v1-decay/step-000171990 \
    --model-config pilots/250m_v1/configs/pilot_250m_decay.yaml \
    --tokenizer-path artifacts/tokenizer_v1.json \
    --packed-corpus-root /workspace/corpus_pilot_train \
    --n-batches 32 \
    --micro-batch 4 \
    --output-json eval-reproduce.json
```

**Expected output:**
```
val_loss : 2.730350
val_ppl  : 15.3383
```

These numbers should be **deterministic** (greedy eval on the same batches). If you get different numbers, something is off — check tokenizer SHA matches `0ad881f58dab…`.

---

## Generate text (~$1-3, interactive)

```bash
# Defaults: 10 smoke-test prompts, temp=0.8, top_p=0.9
python scripts/generate.py \
    --checkpoint /workspace/ckpt/pilot-250m-v1-decay/step-000171990 \
    --model-config pilots/250m_v1/configs/pilot_250m_decay.yaml \
    --tokenizer-path artifacts/tokenizer_v1.json
```

```bash
# Custom prompt
python scripts/generate.py \
    --checkpoint /workspace/ckpt/pilot-250m-v1-decay/step-000171990 \
    --model-config pilots/250m_v1/configs/pilot_250m_decay.yaml \
    --tokenizer-path artifacts/tokenizer_v1.json \
    --prompt "The best programming language is" \
    --max-new-tokens 100 \
    --temperature 0.8 --top-p 0.9
```

```bash
# Greedy / deterministic (always picks highest-prob token)
python scripts/generate.py [...] --greedy
```

Sampling time: ~50-100 ms per token on H100 after JIT warmup (first run is +30-60 sec for compilation).

---

## Full training reproducer (Stage 1 from scratch)

⚠️ This is for "rebuild from scratch" purposes. **For Stage 2 work, use a different config** (Stage 2 needs the 1B `configs/base_1b.yaml` + larger corpus + multi-epoch reader).

### Step A: corpus build (~2 hr on 128-core CPU, ~$0 if dev box)

```bash
# On a 128-core CPU box with .env credentials
cd /workspace/llm-build
source .venv/bin/activate

python scripts/run_parallel_builds.py \
    --pretrain-mix-config pilots/250m_v1/configs/pretrain_mix_pilot.yaml \
    --tokenizer-path artifacts/tokenizer_v1.json \
    --output-root /workspace/corpus_pilot_build/sources \
    --r2-prefix corpus_v1_pilot/sources \
    --target-tokens-per-source 5000000000 \
    --max-parallel 8 \
    --delete-local-after-upload \
    --production
```

Wall: ~2 hr. Output: `s3://llm-data/corpus_v1_pilot/sources/<source-id>/` for 13 sources, ~5 B tokens total.

### Step B: compose pass (~1 h 50 m)

After all 13 sources land in R2, compose them into a single training corpus:

```bash
# Pull the per-source corpora back local (compose script reads local, not R2)
mkdir -p /workspace/corpus_pilot_build/sources
aws --endpoint-url "$S3_ENDPOINT_URL" s3 sync \
    s3://$S3_BUCKET/corpus_v1_pilot/sources/ \
    /workspace/corpus_pilot_build/sources/

# Compose
python scripts/compose_mixed_corpus.py \
    --sources-root /workspace/corpus_pilot_build/sources \
    --output-dir /workspace/corpus_pilot_train \
    --pretrain-mix-config pilots/250m_v1/configs/pretrain_mix_pilot.yaml \
    --sequences-per-shard 65536 \
    --strict-sources \
    --corpus-name corpus_v1_pilot_train

# Upload composed corpus to R2
aws --endpoint-url "$S3_ENDPOINT_URL" s3 sync \
    /workspace/corpus_pilot_train s3://$S3_BUCKET/corpus_v1_pilot/train/
```

Output: 608,088 sequences × 8193 tokens, 10 shards, ~20 GB.

### Step C: Stage 1 pretrain (~12 hr on 4×H200, ~$170)

```bash
# Set env vars (see top of this doc)
# Bring up 4×H200 SXM pod, pull repo + corpus + tokenizer

tmux new -s pilot
python scripts/run_pretrain.py \
    --model-config pilots/250m_v1/configs/pilot_250m.yaml \
    --data-config pilots/250m_v1/configs/pretrain_mix_pilot.yaml \
    --tokenizer-path artifacts/tokenizer_v1.json \
    --packed-corpus-root /workspace/corpus_pilot_train \
    --run-name pilot-250m-v1-reproduce \
    --total-steps 229000 \
    --micro-batch-override 4 \
    --log-every 100 \
    --eval-every 5000 \
    --eval-n-batches 32 \
    --checkpoint-every 5000 \
    --checkpoint-root /workspace/ckpt/pilot-250m-v1-reproduce \
    --checkpoint-r2-prefix checkpoints/pilot-250m-v1-reproduce \
    2>&1 | tee /workspace/pilot-reproduce.log
```

**Expected behavior:**
- Initial loss ~11.81 (random init)
- By step 5K: loss ~4.0
- By step 65K: loss ~2.9
- **By step ~152K: corpus exhausts → `training_complete` fires** (the same single-epoch behavior we hit; remove this constraint via Phase 1.1 multi-epoch reader to actually reach step 229K)

WSD decay was scheduled at step 194,650 (last 15%). With single-epoch corpus, it won't be reached without Stage 1.5 or multi-epoch.

### Step D: Stage 1.5 decay-only continuation (~$32, ~2h 18m)

After Stage 1 stops at ~152K, the decay pass:

```bash
# Pre-stage the checkpoint into a fresh dir
mkdir -p /workspace/ckpt/pilot-250m-v1-decay-reproduce
cp -r /workspace/ckpt/pilot-250m-v1-reproduce/step-000151990 \
       /workspace/ckpt/pilot-250m-v1-decay-reproduce/

# Launch decay-only run
python scripts/run_pretrain.py \
    --model-config pilots/250m_v1/configs/pilot_250m_decay.yaml \
    --data-config pilots/250m_v1/configs/pretrain_mix_pilot.yaml \
    --tokenizer-path artifacts/tokenizer_v1.json \
    --packed-corpus-root /workspace/corpus_pilot_train \
    --run-name pilot-250m-v1-decay-reproduce \
    --total-steps 171990 \
    --micro-batch-override 4 \
    --log-every 100 \
    --eval-every 1000 \
    --eval-n-batches 32 \
    --checkpoint-every 2000 \
    --checkpoint-root /workspace/ckpt/pilot-250m-v1-decay-reproduce \
    --checkpoint-r2-prefix checkpoints/pilot-250m-v1-decay-reproduce \
    --reset-data-position-on-resume \
    2>&1 | tee /workspace/pilot-decay-reproduce.log
```

**Schedule math** for the decay-only run:
- total_steps = 171,990
- decay_fraction = 20000 / 171990 ≈ 0.1163
- Stable: 0 → 151,990 (already done from Step C)
- **Decay**: 151,990 → 171,990 (20K NEW steps, LR walks 3e-4 → 3e-5)

**Expected outcome**: val_loss ≈ 2.730 (matches our pilot result within noise).

---

## Resuming the pilot for inspection (1×anything pod, ~$1)

Just load + look at outputs. Same as "Post-hoc eval" + "Generate text" sections above. The G6 reshard fix (commit `ca1c40b`+) lets you load on any device count.

---

## Sanity checks before launching anything new

```bash
# 1. Verify R2 connectivity
aws --endpoint-url "$S3_ENDPOINT_URL" s3 ls s3://$S3_BUCKET/

# 2. Verify GPU visibility (JAX + torch should both see all GPUs)
python -c "import jax, torch; print('JAX:', jax.devices()); print('torch:', torch.cuda.device_count())"

# 3. Verify the tokenizer matches the pilot SHA (anti-drift safeguard)
python -c "
import hashlib
sha = hashlib.sha256(open('artifacts/tokenizer_v1.json', 'rb').read()).hexdigest()
expected = '0ad881f58dab'  # first 12 chars; full sha in corpus manifest.json
print('OK' if sha.startswith(expected) else 'MISMATCH')
print('  got:     ', sha[:12])
print('  expected:', expected)
"
```

---

## Common gotchas (from this pilot)

| Symptom | Fix |
|---|---|
| `NCCL operation ncclGroupEnd() failed` on multi-GPU launch | `export NCCL_NVLS_ENABLE=0` + `export NCCL_IB_DISABLE=1` before launch |
| `CUDNN_STATUS_NOT_INITIALIZED` at startup | `unset LD_LIBRARY_PATH` before activating venv |
| pip install errors during pod_launch_gpu.sh | Re-run; it's idempotent. Don't add `--force-reinstall` (corrupts nvidia namespace) |
| `OverflowError: Python int ... too large to convert to int32` at step ~65K | Already fixed in commit `9f442f7`; make sure you're on `main` post-fix |
| `sharding passed to deserialization should be specified ... Got None` | Already fixed in commits `13d6126` + `3be12de` + `ca1c40b`; G6 reshard support active |
| `aws s3 sync` "Max Retries Exceeded" on large files | Use the boto3 + 32 MB chunks pattern from `docs/SESSION_HANDOFF_2026-05-14.md` §1 |
| Corpus has only 9 shard dirs after sync | Likely partial shard download. Re-run sync — it's idempotent. Verify each shard has 4 files (tokens.bin + 3 metadata) before training |
| W&B error: `api_key not configured (no-tty)` | `export WANDB_API_KEY=...` or use `--no-wandb` |
| Training stopped at ~step 152K instead of `--total-steps 229K` | Corpus exhausted (5 B tokens / 32,768 per step). Run Stage 1.5 OR (Stage 2+) use multi-epoch reader from Phase 1.1 |

---

## Required CLI flags reference

Adapted from `scripts/run_pretrain.py --help`:

| Flag | Pilot value | What it does |
|---|---|---|
| `--model-config` | `configs/pilot_250m.yaml` (or `_decay.yaml`) | Model architecture spec |
| `--data-config` | `configs/data/pretrain_mix_pilot.yaml` | 13-source data mixture |
| `--tokenizer-path` | `artifacts/tokenizer_v1.json` | Local tokenizer file (auto-downloaded if missing + `--tokenizer-key` set) |
| `--packed-corpus-root` | `/workspace/corpus_pilot_train` | Local path to composed corpus |
| `--run-name` | e.g. `pilot-250m-v1-2026-05-13` | W&B run identifier |
| `--total-steps` | 229000 (Stage 1) or 171990 (Stage 1.5) | Hard step ceiling |
| `--micro-batch-override` | 4 | Per-device batch size override |
| `--log-every` | 100 | Steps between log lines |
| `--eval-every` | 5000 (Stage 1) or 1000 (Stage 1.5) | Steps between val_loss eval runs |
| `--eval-n-batches` | 32 | Held-out batches per eval |
| `--checkpoint-every` | 5000 (Stage 1) or 2000 (Stage 1.5) | Steps between Orbax saves |
| `--checkpoint-root` | `/workspace/ckpt/<run-name>/` | Local checkpoint dir (R2 mirror happens on save) |
| `--checkpoint-r2-prefix` | `checkpoints/<run-name>` | R2 prefix for mirroring |
| `--reset-data-position-on-resume` | (Stage 1.5 only) | Rewind data cursor to 0 on resume; for decay-only continuation passes |
| `--no-wandb` | (smoke tests) | Disable W&B logging entirely |
| `--synthetic-data` | (smoke tests only) | Use random tokens instead of packed corpus |
| `--peak-lr-override` | (rare) | CLI override for lr_schedule.peak_lr |
