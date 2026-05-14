# MyLLM 1B-Class Base Model Card

Status: live draft, refreshed for the pre-2 plan on 2026-05-14.

This card is a release scaffold. It should be regenerated from the final
training manifest, config, eval harness, and checkpoint metadata before any
public checkpoint is published.

## Model Details

- Model name: MyLLM-1B-Base, final release name TBD
- Developed by: Samatva
- Model type: decoder-only causal language model
- Target scale: about 1B active parameters for the dense pre-2 foundation run
- Planned training budget: at least 1T tokens; pre-2 plan targets 1.5T tokens
- Target context: 8k pretrain context with a later long-context extension path
- License: TBD after legal review; permissive release is the target

## Architecture Direction

The current pre-2 decision is a dense Transformer foundation model, not an MoE
production run. MoE remains a research branch because at 1B scale the routing,
load-balancing, kernel, and serving complexity is not justified until the dense
baseline is strong and reproducible.

Planned architectural ingredients:

- Pre-norm decoder-only Transformer
- RMSNorm
- SwiGLU feed-forward blocks
- RoPE positional encoding
- Grouped-query attention if profiling confirms the quality/speed tradeoff
- BF16 training weights and activations
- TorchTitan-style distributed training stack with PyTorch FSDP2 and
  distributed checkpointing for the pre-2 redesign

The authoritative architecture and training decision record is
[`../PRE2_FINAL_PLAN_2026-05-14.md`](../PRE2_FINAL_PLAN_2026-05-14.md).

## Intended Use

- Internal foundation-model research and stack validation
- Continued pretraining, evaluation, and post-training experiments
- Not intended as a production assistant without safety tuning, eval gates,
  monitoring, and deployment controls

## Out-Of-Scope Use

- Medical, legal, financial, or safety-critical decisions
- Unreviewed production deployment
- Generating or processing private data without DPDP/GDPR-style controls
- Autonomous tool use without a separate agent-safety review

## Training Data

The pre-2 data target is a deduplicated, documented, license-reviewed corpus of
at least 1T tokens, with a planning target of 1.5T tokens.

Candidate source families include:

- High-quality web and educational web data
- Code data with explicit terms review
- Wikipedia and other reference material with attribution/share-alike review
- Public-domain books
- Scientific text
- Math text
- Stack Exchange-style Q&A, capped and legally reviewed
- Indic and multilingual sources with explicit language shares

The final data card must use emitted token counts from the build manifest, not
planning percentages.

## Training Methodology

Planned pre-2 methodology:

- Large-scale autoregressive next-token pretraining
- BF16 mixed-precision training
- Stable distributed checkpointing and deterministic resume tests
- Validation, benchmark, and contamination checks at fixed intervals
- No locked hetero-tokenizer top-K teacher-distillation plan for the base run
- Quantized variants produced after training for inference, not 4-bit/8-bit
  foundation training

## Quantization And Precision

The trainable base model should be trained in BF16. INT8/FP8 may be used only
where the training stack and hardware support it safely, such as optimizer,
communication, or selected matmul paths after profiling. INT4 is an inference
compression format, not the default pretraining precision for this program.

Post-training deliverables should include:

- BF16 or FP16 reference checkpoint
- INT8 inference variant
- Weight-only INT4 inference variant after quality regression testing

## Evaluation Plan

Release evaluation should include:

- General reasoning: MMLU-Pro and related held-out suites
- Multilingual reasoning: MMLU-ProX, Belebele, MILU
- Code: HumanEval+, MBPP+, LiveCodeBench
- Math: GSM8K, MATH-style sets, MGSM
- Long context: RULER or equivalent synthetic probes
- Safety: toxicity, refusal, bias, PII memorization, and jailbreak probes
- Memorization: verbatim extraction and benchmark-contamination checks
- Calibration: expected calibration error by task family

Every reported result must include benchmark version, harness commit, prompt
format, decoding settings, contamination policy, and confidence intervals where
appropriate.

## Limitations

- A 1B-class model trained on 1T to 1.5T tokens is a foundation baseline, not a
  frontier assistant.
- Hindi and broader Indic quality will depend heavily on final source quality
  and token share.
- Code quality depends on license-clean code data and contamination discipline.
- The base model will not be instruction-following without post-training.
- Quantized inference checkpoints may regress reasoning, math, or multilingual
  quality unless calibrated and evaluated.

## Environmental Impact

To be filled from the actual run:

- GPU type and count
- Training duration
- Energy estimate
- Provider region and carbon-intensity estimate where available
- Failed-run and restart overhead

## References

- Current plan: [`../PRE2_FINAL_PLAN_2026-05-14.md`](../PRE2_FINAL_PLAN_2026-05-14.md)
- Architecture decision: [`../PRE2_ARCHITECTURE_DECISION.md`](../PRE2_ARCHITECTURE_DECISION.md)
- Stack migration: [`../PRE2_STACK_MIGRATION_PLAN.md`](../PRE2_STACK_MIGRATION_PLAN.md)
- Project overview: [`../PROJECT_OVERVIEW.md`](../PROJECT_OVERVIEW.md)
- Data card: [`data_card_v1.md`](data_card_v1.md)
- License register: [`license_register.md`](license_register.md)
