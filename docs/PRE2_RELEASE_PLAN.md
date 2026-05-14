# Pre-2 Release And Launch Plan

Date: 2026-05-14
Status: planning contract

## Decided Mainline

The pre-2 mainline is a dense decoder-only Transformer, not MoE.

| Decision | Value |
| --- | --- |
| Main checkpoint family | `myllm-pre2-dense-1.5b-base` |
| Estimated parameters | 1,484,874,240 |
| Public shorthand | 1.5B-class dense model |
| Training token minimum | 1.0T tokens |
| Base release target | 1.5T tokens |
| Stretch continuation | 3.0T tokens if proxy curves, evals, data quality, and budget justify it |
| Foundation context | 8,192 tokens |
| Long-context continuation | 16k/32k after base stability, not the whole run |
| Canonical training dtype | BF16 |
| Quantized variants | Export/eval after BF16 base; not 4-bit or 8-bit base training |

This is a working decision for implementation. It can still be falsified by proxy runs, tokenizer studies, data quality, or hardware economics.

## Compute-Limited POC Ladder

Because training compute is shared with other model runs, architecture validation should climb a smaller ladder before the 1.5B target.

| Stage | Config | Estimated params | Token target | Why it exists |
| --- | --- | ---: | ---: | --- |
| Canary | `configs/pre2_dense_canary_110m.yaml` | 112,737,280 | 1B | Fast failure detection for model, trainer, data, checkpoint, and eval bridge |
| POC | `configs/pre2_dense_poc_250m.yaml` | 239,104,256 | 10B | Meaningful loss/eval/data signal under limited compute |
| Proxy | `configs/pre2_dense_proxy_400m.yaml` | 380M-class | 30B | Stronger scale signal before approving 1.5B |
| Mainline | `configs/pre2_dense_1_5b.yaml` | 1,484,874,240 | 1.0T-1.5T | v1 base release candidate |

This ladder changes launch sequencing, not the final architecture target. If the 110M or 250M runs fail to show stable loss, exact resume, and real eval movement, we should not spend the 400M or 1.5B compute yet.

## Launch Stages

| Stage | Audience | Model scope | Token scope | Hard gate |
| --- | --- | --- | --- | --- |
| v0.1-pre2 | Internal developers | Tiny smoke only | synthetic + tiny packed fixtures | config, loss-mask, BF16 smoke, checkpoint resume pass |
| v0.2-canary | Internal research | 110M canary | 0.5B-1B study tokens | real eval bridge smoke, stable loss, exact resume |
| v0.3-poc | Internal research | 250M POC | 3B-10B study tokens | source-bucket metrics, eval movement, data/tokenizer sanity |
| v0.3-data | Internal data review | no launch model | corpus manifests only | source registry, license manifest, dedup, decontam, tokenizer ablation |
| v0.4-proxy | Internal training | 400M proxy | 10B-30B on target stack | 8-GPU training, DCP restore, throughput, BF16 stability |
| v0.5-internet-preview | Gated external preview | best validated POC/proxy checkpoint, no public weights | likely 10B-30B unless compute frees up | safety filters, monitoring, eval card, data/model/risk cards |
| v1-internal | Internal base release | 1.5B dense base | at least 1.0T, target 1.5T | full eval scorecard, contamination report, exact resume, governance packet |
| v1-public | Public/API or weights decision | post-trained/safety-reviewed model | usually 1.5T+ base plus post-train | legal clearance, safety review, abuse monitoring, reproducible cards |

## Why v0.5 Is Not v1

v0.5 should be a gated internet preview, not a public foundation release. It proves product, serving, monitoring, and safety workflow under controlled access. It should not imply that the base model is final, that weights are releasable, or that the data corpus is complete.

## Go/No-Go Gates

### v0.2 Canary Gate

- 110M canary trains without recurring NaNs.
- Checkpoint restore produces the same next-step behavior as uninterrupted training.
- Packed-corpus path consumes aligned labels and loss masks.
- Real eval bridge runs at least a toy checkpoint end to end.
- Tokenizer candidates have fertility and throughput reports.

### v0.3 POC Gate

- 250M POC reaches 3B useful tokens minimum, 10B target if curves are stable.
- Validation loss improves smoothly and source-bucket loss does not collapse.
- Initial eval movement is positive enough to justify 400M proxy compute.
- Data manifests exist for every POC source shard.

### v0.4 Scale Gate

- TorchTitan/FSDP2/DCP or fallback runtime is selected by measured throughput and resume behavior.
- 8-GPU interrupt/resume drill passes.
- BF16 training is stable for a 10B-30B-token 400M proxy run.
- Data source manifests are complete for the canary subset.
- Eval movement is positive enough to justify 1.5B main run.

### v1 Internal Gate

- At least 1.0T useful tokens are trained.
- 1.5T target is reached unless eval/loss curves clearly saturate or budget gate stops the run.
- Full release eval scorecard runs from a real checkpoint.
- Data card, model card, risk card, license register, and contamination report are generated from artifacts.
- Serving/export path works for BF16 and at least one validated quantized export.

## Current Validation Status

| Area | Status |
| --- | --- |
| Dense config and parameter math | validated by `scripts/pre2_config_check.py` and tests |
| Token budget contract | 1.0T minimum, 1.5T release target, 3.0T stretch |
| Local model/loss | smoke validated |
| Packed-corpus loss mask | smoke validated |
| BF16 smoke | local autocast smoke validated |
| Single-process checkpoint resume | exact test validated |
| 110M/250M compute-limited configs | planning configs and parameter math validated |
| Distributed TorchTitan training | not implemented |
| DCP distributed resume | not implemented |
| Source registry / license gate | initial planning registry implemented; v0.5 POC sources approved, mainline sources still require approval |
| v0.5 data/source readiness | FineWeb-Edu and OpenWebMath pinned and approved for internal POC/proxy; tokenizer/decontamination artifacts staged on VM and mirrored to `llm-data-rust` |
| v0.5 corpus build commands | build-time HF streaming config and per-source command emitter added |
| CPU data-prep VM | Ubuntu VM with 64 CPU cores, 83GiB RAM, 800GiB root disk, `/data/pre2/v0_5` layout, and pre-2 tests passing |
| Tiny checkpoint eval bridge | next-token token-ID eval implemented |
| Full release eval bridge | not implemented |
| Internet preview serving/safety | not implemented |

## Immediate Next Engineering Gates

1. Run a bounded v0.5 corpus build smoke on the CPU data-prep VM.
2. Build the full 10B-token v0.5 packed corpus or a smaller canary corpus if wall-clock risk appears high.
3. Full lm-eval/LightEval adapter over the pre-2 checkpoint predict function.
4. TorchTitan adapter skeleton.
5. 110M canary train/eval loop.
6. 250M POC train/eval loop.
7. 8-GPU checkpoint/resume drill.

## Reviewer Entry Points

Ask reviewers to start with:

- `docs/PRE2_REVIEWER_QUESTIONS_2026-05-14.md`
- `docs/PRE2_ARCHITECTURE_DECISION.md`
- `docs/PRE2_FINAL_PLAN_2026-05-14.md`
- `docs/PRE2_RELEASE_PLAN.md`
- `configs/pre2_dense_poc_250m.yaml`
- `configs/pre2_dense_1_5b.yaml`
- `configs/data/pre2_source_registry.yaml`
- `configs/data/pre2_mix_v0_5.yaml`
