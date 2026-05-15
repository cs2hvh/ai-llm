# MyLLM Pilot 250M v1 — R2 Inventory

All durable artifacts for the pilot. R2 is the canonical store; pod local disks were ephemeral.

**Bucket**: `s3://llm-data/` (Cloudflare R2, account-specific endpoint in `.env`)
**Total pilot-related storage on R2**: ~146 GB (out of 161 GB total bucket usage)
**Verified inventory date**: 2026-05-15

## Final model — the artifact for the model card

```
s3://llm-data/checkpoints/pilot-250m-v1-decay/step-000171990/
├── manifest.json         (92 B,  "reason": "final", step: 171990)
├── eval-final-decay.json (180 B, val_loss: 2.7303, val_ppl: 15.34)
└── state/
    └── (Orbax sharded state, 18 files total, ~2.65 GB)

Last modified: 2026-05-15T01:42:59Z
```

Pull command:
```bash
mkdir -p /workspace/ckpt/pilot-250m-v1-decay
aws --endpoint-url "$S3_ENDPOINT_URL" s3 sync \
    s3://$S3_BUCKET/checkpoints/pilot-250m-v1-decay/step-000171990/ \
    /workspace/ckpt/pilot-250m-v1-decay/step-000171990/
```

## Stage 1 final (pre-decay) — for comparison

```
s3://llm-data/checkpoints/pilot-250m-v1/step-000151990/
├── manifest.json
├── eval-final.json       (172 B, val_loss: 2.8776, val_ppl: 17.77)
└── state/                (Orbax sharded state, ~2.65 GB)
```

## All Stage 1 + Stage 1.5 checkpoints (full lineage)

### Stage 1 — every 5000 steps (31 checkpoints, ~82 GB)

```
s3://llm-data/checkpoints/pilot-250m-v1/
├── step-000005000/   step-000010000/   step-000015000/
├── step-000020000/   step-000025000/   step-000030000/
├── step-000035000/   step-000040000/   step-000045000/
├── step-000050000/   step-000055000/   step-000060000/
├── step-000065000/   step-000070000/   step-000075000/
├── step-000080000/   step-000085000/   step-000090000/
├── step-000095000/   step-000100000/   step-000105000/
├── step-000110000/   step-000115000/   step-000120000/
├── step-000125000/   step-000130000/   step-000135000/
├── step-000140000/   step-000145000/   step-000150000/
└── step-000151990/   ← Stage 1 FINAL (corpus exhausted here)
```

Each ~2.65 GB. Useful if you need to study mid-run state (e.g., loss curve checkpoints, watchdog rollback recovery experiments).

### Stage 1.5 — every 2000 steps (11 checkpoints, ~29 GB)

```
s3://llm-data/checkpoints/pilot-250m-v1-decay/
├── step-000152000/   step-000154000/   step-000156000/
├── step-000158000/   step-000160000/   step-000162000/
├── step-000164000/   step-000166000/   step-000168000/
├── step-000170000/
└── step-000171990/   ← STAGE 1.5 FINAL (the headline model)
```

These trace the WSD decay phase: LR walks 3e-4 → 3e-5 across these 20K steps.

## Composed pilot corpus (training data)

```
s3://llm-data/corpus_v1_pilot/train/
├── manifest.json         (top-level — corpus_v1_pilot_train)
└── shard-000000/ ... shard-000009/   (10 shards, ~20 GB total)
    └── (each shard has: tokens.bin, seq_meta.arrow, doc_meta.parquet, manifest.json)

Total: 41 files, 20.09 GB
Sequences: 608,088
Total tokens: 4,982,022,257
Sequence length: 8193
Tokenizer SHA: 0ad881f58dab... (matches artifacts/tokenizer_v1.json)
Build timestamp: 2026-05-13T21:54:25Z
Source-share drift: 0.33% max (under 2% L5 threshold — passes)
```

Per-source share (target → actual):

```
fineweb_edu              44.0% → 44.15%   (+0.15 pp)
github_code_clean        18.0% → 18.06%   (+0.06 pp)
open_web_math             7.0% →  7.02%   (+0.02 pp)
pes2o                     6.0% →  6.02%   (+0.02 pp)
wikipedia_20231101_en     6.0% →  6.02%   (+0.02 pp)
pg19                      5.0% →  4.67%   (-0.33 pp)  ← pg19 corpus is finite
sangraha_verified_split_hin 4.0% → 4.01% (+0.01 pp)
stack_exchange_preferences 2.0% → 2.01%   (+0.01 pp)
mc4_de                    2.0% →  2.01%   (+0.01 pp)
mc4_ar                    1.5% →  1.51%   (+0.01 pp)
mc4_es                    1.5% →  1.51%   (+0.01 pp)
mc4_fr                    1.5% →  1.51%   (+0.01 pp)
mc4_zh                    1.5% →  1.51%   (+0.01 pp)
```

pg19 (books) ran out of source data — expected, documented in `docs/pilot_corpus_rebuild_plan.md`.

## Per-source corpora (build outputs)

```
s3://llm-data/corpus_v1_pilot/sources/
├── fineweb_edu/                       (5 shards,    2.20 B tokens)
├── github_code_clean/                 (2 shards,    900 M tokens)
├── open_web_math/                     (1 shard,     350 M tokens)
├── pes2o/                             (1 shard,     300 M tokens)
├── wikipedia_20231101_en/             (1 shard,     300 M tokens)
├── pg19/                              (1 shard,     232 M tokens — capped)
├── sangraha_verified_split_hin/       (1 shard,     200 M tokens)
├── stack_exchange_preferences/        (1 shard,     100 M tokens)
├── mc4_de/                            (1 shard,     100 M tokens)
└── mc4_{ar,es,fr,zh}/                 (1 shard each, 75 M tokens each)

Total: 85 files, 20.08 GB
```

These are the per-source corpora BEFORE compose. The composed `train/` corpus is built from these via deficit-driven sampling.

## Tokenizer

```
s3://llm-data/tokenizer/myllm-spm-unigram-131k-v2.json
  Size: 4.79 MB
  Vocab: 131,072
  Type: SentencePiece Unigram
  Special tokens: BOS, EOS, PAD, UNK, IM_START, IM_END, TOOL_CALL, TOOL_RESULT
  SHA: 0ad881f58dab... (full sha embedded in corpus manifest)
  Built: 2026-05-11T09:40:45Z
```

## Decontamination indexes

```
s3://llm-data/decontamination/
├── decontamination_index_8gram.json     (37.52 MB,  1.75 M ngrams)
└── decontamination_index_13gram.json    (37.21 MB,  1.74 M ngrams)
```

Built from 10 benchmarks: bbh, gsm8k, humaneval-plus, ifeval, math, mbpp-plus, mgsm, mmlu-pro, mmlu-prox, belebele. MILU is gated on HF; harshit.hv needs to request access at [huggingface.co/datasets/ai4bharat/MILU](https://huggingface.co/datasets/ai4bharat/MILU) before it can be added.

Note: 558 docs from codeparrot/github-code-clean were caught by the dual-mode decontam during the corpus build — verified real catch, not over-filtering.

## Summary table

| Category | Path | Size | Files |
|---|---|---|---|
| **Final model checkpoint** | `checkpoints/pilot-250m-v1-decay/step-000171990/` | 2.65 GB | 18 |
| Stage 1 final | `checkpoints/pilot-250m-v1/step-000151990/` | 2.65 GB | 18 |
| All Stage 1 checkpoints | `checkpoints/pilot-250m-v1/` | ~82 GB | 31 dirs |
| All Stage 1.5 checkpoints | `checkpoints/pilot-250m-v1-decay/` | ~29 GB | 11 dirs |
| Composed corpus | `corpus_v1_pilot/train/` | 20.09 GB | 41 |
| Per-source corpora | `corpus_v1_pilot/sources/` | 20.08 GB | 85 |
| Tokenizer | `tokenizer/myllm-spm-unigram-131k-v2.json` | 4.79 MB | 1 |
| Decontam indexes | `decontamination/` | 75 MB | 2 |
| **Total pilot-related** | — | **~146 GB** | ~190 objects |

## Things NOT on R2 (no backup)

- Training logs (`pilot.log`, `pilot-decay.log`) — pod was torn down before upload; reconstruct via W&B if needed
- Quarantine log (`quarantine.jsonl`) — same situation
- W&B run state — lives in W&B cloud, not R2. Runs `roydqofb` (pre-crash) + `u5xsxm0l` (post-resume) + `pxoungh9` (decay pass) all finalized.

For forensic debugging, W&B has the curves + system metrics. Logs would only matter if we hit a future regression and wanted to compare event-by-event.
