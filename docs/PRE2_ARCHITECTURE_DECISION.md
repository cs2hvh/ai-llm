# Pre-2 Architecture Decision Record

Date: 2026-05-14
Status: accepted for pre-2 implementation

## Context

The pre-1 stack proved useful parts of the system: tokenizer work, packed
corpus mechanics, canaries, checkpoint lessons, and pilot-scale training.
It is not the right production stack for the next goal: a 1B-class foundation
model trained on at least 1T tokens, with a target of 1.5T tokens and a path
to 3T.

The pre-2 decision must optimize for reproducibility, distributed training
correctness, data quality, and a credible release path. It should not optimize
for novelty before a dense baseline exists.

## Decision

Pre-2 mainline is a dense decoder-only Transformer:

| Field | Decision |
|---|---:|
| Model name | `myllm-pre2-dense-1.5b-base` |
| Total parameters | about 1.49B |
| Layers | 20 |
| Hidden size | 2048 |
| FFN size | 8192 |
| Attention | GQA, 32 query heads, 8 KV heads |
| Head dim | 64 |
| Norm | RMSNorm, pre-norm |
| Activation | SwiGLU |
| Stabilization | QK-norm, z-loss |
| Position | RoPE |
| Vocab | tokenizer ablation: 64k, 96k, 131k |
| Embeddings | tied |
| Foundation context | 8192 |
| Long-context target | 32768 via continuation |
| Training dtype | BF16 baseline |
| Optimizer | AdamW baseline |
| Schedule | WSD |
| Stack | PyTorch/TorchTitan/FSDP2/DTensor/DCP |

Parameter estimate:

```text
embedding = 131075 * 2048 = 268,441,600
attention_per_layer = 10,485,760
ffn_per_layer = 50,331,648
20 layers ~= 1,216,430,080 non-embedding params
total ~= 1,484,865,536 plus norms and small terms
```

## Rejected For Mainline

MoE is rejected as the pre-2 default. A 1B-total MoE does not buy enough
capacity to justify routing, load balancing, expert parallelism, grouped GEMM,
and serving complexity. MoE remains a research branch only after the dense
baseline is stable.

Cross-tokenizer top-K logit distillation is rejected. Teacher token IDs are
not student token IDs when tokenizers differ. Use teacher-generated text,
reasoning traces, same-tokenizer KD, or explicit cross-tokenizer KD research
methods only.

Direct 4-bit or 8-bit base pretraining is rejected. Train the canonical model
in BF16 first. FP8 training can be tested after the baseline is stable. INT8
and INT4 are deployment artifacts produced from a trained checkpoint.

Muon, MTP, MLA, and other advanced changes are ablations, not baseline
requirements. Add one new risk axis at a time after the dense baseline works.

## Consequences

This decision creates these hard engineering requirements:

1. Pre-2 training code must be PyTorch-first, not a patch on the Keras/JAX trainer.
2. Packed corpus loading must support exact resume by sequence id and manifest hash.
3. Checkpointing must use distributed checkpointing and pass restore tests across process counts.
4. Eval must use real model predictions before any release claim.
5. Tokenizer identity must be stored in corpus manifests, checkpoints, eval outputs, and scorecards.
6. Governance docs must be generated from actual manifests and run metadata, not planning estimates.

## Config Artifacts

- `configs/pre2_dense_1_5b.yaml`
- `configs/pre2_dense_proxy_400m.yaml`
- `configs/pre2_moe_1b_active_research.yaml`
- `configs/data/pre2_mix_stage1.yaml`
- `configs/data/pre2_mix_stage2.yaml`
- `configs/data/pre2_mix_anneal.yaml`

These are planning configs until the TorchTitan adapter and pre-2 dataloader
exist. They must not be passed to the pre-1 JAX/Keras entrypoints.
