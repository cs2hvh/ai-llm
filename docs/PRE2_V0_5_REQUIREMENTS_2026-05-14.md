# Pre-2 v0.5 24-Hour Launch Requirements

Date: 2026-05-14
Status: planning contract

## Goal

v0.5 is a gated internet preview from the compute-limited pre-2 ladder. It is
not the v1 foundation release and should not ship public weights. The goal is
to prove that the architecture, data path, checkpointing, eval bridge, serving
path, and safety/monitoring loop work under real usage pressure.

## Model Scope

Default v0.5 target:

| Item | Decision |
| --- | --- |
| Config | `configs/pre2_dense_poc_250m.yaml` |
| Model size | 239,104,256 parameters with the current 131,075 runtime vocab contract |
| Training budget | 10B useful tokens target, 3B minimum |
| Context | 8,192 tokens |
| Precision | BF16 training; post-training 8-bit or 4-bit export only |
| Audience | gated external preview |

Stretch target, only if 8 GPUs are available and data is already packed:

| Item | Decision |
| --- | --- |
| Config | `configs/pre2_dense_proxy_400m.yaml` |
| Model size | 377M-class parameters |
| Training budget | 30B useful tokens |
| Audience | stronger v0.5 preview candidate |

## Dataset Decision

For a 24-30 hour v0.5 window, use a two-source mix first. This reduces
ingestion, licensing, dedup, and provenance risk while still giving a real
general-plus-reasoning signal.

| Source | v0.5 share | Token target for 10B run | Why |
| --- | ---: | ---: | --- |
| FineWeb-Edu pinned slice | 85% | 8.5B | high-quality educational web; strong baseline corpus |
| OpenWebMath pinned slice | 15% | 1.5B | math/STEM density and cleaner reasoning signal |

Do not include code, StackExchange, or Indic sources in the first 24-hour v0.5
unless their source revision, terms, PII controls, and decontamination reports
are already ready before the run starts. They are good v0.5.1 additions, but
they add operational risk to the first launch.

Required source controls before POC corpus build:

- Pin exact dataset revision or snapshot.
- Record license expression and terms URL.
- Store per-document hashes and source IDs in the packed-corpus metadata.
- Run dedup and benchmark decontamination reports.
- Run PII/secrets filtering for all web/code-like material.
- Record tokenizer SHA-256 in the corpus manifest.

## Tokenizer Decision

Current repo state:

- The tokenizer spec exists at `configs/tokenizer.yaml`.
- The expected tokenizer name is `myllm-spm-unigram-131k-v2`.
- The expected remote object key is `tokenizer/myllm-spm-unigram-131k-v2.json`.
- Pod scripts pull it into `artifacts/tokenizer_v1.json`.
- No local `artifacts/tokenizer_v1.json` artifact is present in this checkout.

Preferred v0.5 path:

1. Pull the existing 131k SentencePiece Unigram tokenizer from object storage.
2. Verify SHA-256 against a manifest before corpus packing.
3. Keep the 250M/400M configs on the same tokenizer family as the 1.5B target.

Fallback only if the remote tokenizer is unavailable at launch start:

1. Train a temporary 64k or 96k SentencePiece Unigram tokenizer from the v0.5
   source sample.
2. Mark the checkpoint as tokenizer-incompatible with the 1.5B mainline.
3. Use it only to prove trainer/data/serving mechanics, not scale transfer.

## GPU Requirement

The requirement below assumes the run must include data staging sanity checks,
training, checkpoint restore, eval, export, and serving smoke inside 24-30 hours.
Training-only can finish faster if the packed corpus already exists.

| Target | Minimum credible GPU | Recommended GPU | Notes |
| --- | --- | --- | --- |
| 250M / 10B v0.5 | 2x H100/H200 if packed data is ready | 4x H100 80GB or 4x H200 141GB | best fit for a reliable 24-hour preview cycle |
| 400M / 30B stretch | 4x H100/H200 with risk | 8x H100/H200 | use only if data is prepacked and cluster time is stable |
| 400M / 30B on Blackwell | 1x 8-GPU DGX B200 class node | 1x 8-GPU DGX B200 class node | overkill for 250M; useful if available |
| A100 fallback | 8x A100 80GB for 250M/10B | reduce to canary if fewer GPUs | acceptable for mechanics, weaker for 24-hour confidence |

Decision: request 4x H100/H200 for the default 250M/10B v0.5. Request 8x
H100/H200 only if the target is upgraded to 400M/30B.

## Storage Requirement

The packed corpus stores token IDs as uint32 because the current tokenizer has
a 131,075 runtime vocabulary with max token ID 131,074. That is 4 bytes per
token before metadata.

| Target | Packed token storage | Local NVMe minimum | Comfortable local NVMe | Object storage |
| --- | ---: | ---: | ---: | ---: |
| 250M / 10B | about 40GB tokens plus metadata | 500GB | 1TB | 1TB |
| 400M / 30B | about 120GB tokens plus metadata | 1.5TB | 2TB | 2-3TB |

Budget details:

- Raw/download cache for a 10B-token two-source run: plan 100-250GB.
- Packed tokens for 10B: 40GB plus Arrow/Parquet metadata and manifests.
- Checkpoints for 250M with optimizer state: plan 4-6GB each; keep at least
  10 checkpoints plus best/export copies.
- Eval outputs, logs, tokenizer sample, and safety telemetry: reserve 50-100GB.

## Go/No-Go

Go for v0.5 only if:

- The tokenizer artifact is present or a fallback tokenizer decision is recorded.
- At least FineWeb-Edu and OpenWebMath have pinned revisions and accepted terms.
- Packed corpus manifests include tokenizer hash, source revisions, and token
  counts.
- A checkpoint restore drill passes before long training.
- Eval runs from a real checkpoint.
- Serving runs behind a gated preview with logs and kill switch.

No-go if:

- Source registry still has zero approved POC sources.
- Tokenizer hash is unknown.
- Data pack lacks provenance.
- Checkpoint resume cannot be demonstrated.
- The only available GPU budget is below the 110M canary threshold.

Machine-check the current gate with:

```bash
python scripts/pre2_v0_5_readiness.py
```

As of the VM setup pass, the data/source side is ready for the POC gate. The
tokenizer and decontamination artifacts were pulled from `llm-data`, staged on
the CPU data-prep VM, and mirrored into `llm-data-rust` under the expected keys.

After the tokenizer and decontamination indexes are staged, emit the exact
per-source corpus build commands with:

```bash
python scripts/pre2_v0_5_build_commands.py
```

The build-time HF streaming config is `configs/data/pre2_v0_5_build_mix.yaml`;
it pins FineWeb-Edu to the `sample-10BT` config and caps sources at 8.5B and
1.5B tokens respectively.

## References To Verify During Source Approval

- FineWeb-Edu: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
- OpenWebMath: https://huggingface.co/datasets/open-web-math/open-web-math
- NVIDIA H100: https://www.nvidia.com/en-us/data-center/h100/
- NVIDIA H200: https://www.nvidia.com/en-au/data-center/h200/
- NVIDIA DGX B200: https://www.nvidia.com/en-us/data-center/dgx-b200/
