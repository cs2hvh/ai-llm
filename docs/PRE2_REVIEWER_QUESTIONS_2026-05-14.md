# Pre-2 Reviewer Questions And Doubts

Date: 2026-05-14
Audience: external AI research / systems reviewer
Status: review packet

## Context For Reviewer

The project is being moved from a pre-1 experimental codebase toward a pre-2
foundation-model training path. The current selected mainline is a dense
decoder-only Transformer, not MoE, with a 1.5B-class target trained on at least
1.0T tokens and preferably 1.5T tokens before an internal v1 base decision.

Because compute is constrained, the immediate launch path is a v0.5 gated
preview built from a smaller POC:

| Stage | Params | Tokens | Purpose |
| --- | ---: | ---: | --- |
| Canary | 112,737,280 | 1B | fast failure detection |
| v0.5 POC | 239,104,256 | 10B | architecture/data/eval signal under 24-30h constraint |
| Proxy | 377,525,248 | 30B | stronger scale signal before 1.5B |
| Mainline | 1,484,874,240 | 1.0T-1.5T | internal v1 base candidate |

Current v0.5 data decision is intentionally simple: FineWeb-Edu `sample-10BT`
at 85 percent and OpenWebMath at 15 percent, with pinned revisions, tokenizer
hashing, decontamination indexes, and packed-corpus provenance.

## Review Entry Points

- `docs/PRE2_ARCHITECTURE_DECISION.md`
- `docs/PRE2_FINAL_PLAN_2026-05-14.md`
- `docs/PRE2_RELEASE_PLAN.md`
- `docs/PRE2_V0_5_REQUIREMENTS_2026-05-14.md`
- `configs/pre2_dense_poc_250m.yaml`
- `configs/pre2_dense_proxy_400m.yaml`
- `configs/pre2_dense_1_5b.yaml`
- `configs/data/pre2_source_registry.yaml`
- `configs/data/pre2_mix_v0_5.yaml`
- `src/myllm_pre2/`
- `scripts/pre2_*`

## Decisions We Want Challenged

1. Dense baseline over MoE for the mainline.
   - Is dense 1.5B still the right default for this team and infra?
   - Would a 1B-active MoE branch provide useful evidence early enough to matter?
   - Are we over-penalizing MoE because of serving and infra complexity?

2. v0.5 model scale.
   - Is 239M parameters over 10B tokens enough to validate architecture/data choices?
   - Should v0.5 target the 377M/30B proxy instead, even if it requires 8 GPUs?
   - What eval movement would justify climbing from 250M to 400M?

3. Tokenizer.
   - Current artifact is SentencePiece-Unigram with byte fallback and a 131,075 runtime vocab.
   - It is acceptable for v0.5, but embeddings dominate small models.
   - Should v1 freeze this tokenizer, or require a 64k/96k/131k tokenizer ablation first?
   - Are there known failure modes with the artifact's special-token layout, including trailing `<s>`, `</s>`, and `<unk>` IDs?

4. Data mix.
   - Is the 85 percent FineWeb-Edu / 15 percent OpenWebMath v0.5 mix too narrow?
   - Should code or multilingual data be included in v0.5 despite added license and pipeline risk?
   - Is OpenWebMath at 15 percent excessive for a general base POC?
   - What source-bucket validation would make the result interpretable?

5. Objective and schedule.
   - Current default is next-token CE, z-loss, AdamW, WSD schedule, BF16.
   - Are WSD parameters reasonable for the 250M/10B and 1.5B/1.5T cases?
   - Should we introduce auxiliary losses, data curricula, or staged anneals earlier?

6. Context length.
   - Foundation context is 8,192 tokens, with 16k/32k continuation later.
   - Is 8k too expensive for v0.5 POC and 1.5B base?
   - Would a shorter POC context give faster and cleaner architecture signal?

7. Evaluation.
   - Current implemented eval is only a tiny checkpoint next-token bridge.
   - Planned evals include MMLU-Pro/ProX, MILU, Belebele, GSM8K/MGSM, MATH, HumanEval+/MBPP+, BBH, IFEval, and source-bucket perplexity.
   - Which eval subset should gate 110M, 250M, 400M, and 1.5B separately?
   - How should contamination reports affect go/no-go decisions?

8. Training stack.
   - Plan currently points to PyTorch/TorchTitan/FSDP2/DCP for the production path.
   - Should we benchmark Megatron-Core or DeepSpeed before committing?
   - What minimum 8-GPU proof is enough before running the 250M POC?

9. Governance and release posture.
   - v0.5 is gated preview only; no public weights.
   - Are the current source registry and license gates strict enough?
   - What extra privacy, provenance, or misuse controls should be mandatory before external preview?

## Specific Doubts

- The 250M POC may be too embedding-heavy with a 131k tokenizer, making it a weaker proxy for the 1.5B architecture than its parameter count suggests.
- The two-source v0.5 corpus is operationally clean but may not reveal code, multilingual, or instruction-following weaknesses early enough.
- Single-process smoke trainer is useful but does not prove distributed throughput, DCP restore, or NCCL failure handling.
- Current packed-corpus build path relies on existing pre-1 data tooling; it needs a hard review for scale and exact provenance before 1T+.
- Current docs are planning contracts plus smoke proofs, not proof of training quality.

## Requested Reviewer Output

Please provide:

1. Keep/change decision for dense 1.5B mainline.
2. Keep/change decision for 250M/10B v0.5.
3. Tokenizer recommendation for v0.5 and v1.
4. Minimum eval suite for each stage.
5. Highest-risk architectural or data assumption.
6. Compute/storage recommendation for v0.5 and 1.5B.
7. Any hard no-go item before spending GPU time.
