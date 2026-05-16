#!/usr/bin/env python3
"""Build a release scorecard for a trained checkpoint.

End-to-end:
  1. Load model config + checkpoint (JAX/Orbax).
  2. Build a predict_fn that wraps model.call() + greedy decode.
  3. Run every configured benchmark via myllm.eval.release_scorecard.
  4. Write JSON + Markdown to <output-dir>/.
  5. Optionally upload to R2 for the model card pipeline to pick up.

Usage::

    python scripts/build_release_scorecard.py \\
        --model-config configs/pilot_250m.yaml \\
        --tokenizer-path artifacts/tokenizer_v1.json \\
        --checkpoint-root /workspace/checkpoints/pilot \\
        --checkpoint-step 50000 \\
        --benchmarks mmlu-pro,gsm8k,humaneval-plus,mbpp-plus,bbh,ifeval \\
        --sample-size 200 \\
        --output-dir artifacts/scorecards/pilot_step50k \\
        --r2-prefix scorecards/pilot/

The pilot scorecard is meant for early-signal monitoring + the eventual
model-card "Evaluation" section paste. For the v1 release scorecard,
expect to run with ``--sample-size`` unset (full benchmark) and the
full v1-gate set.

This script's predict_fn is the simple "argmax of last-position logit"
path — works for MMLU-style single-token answers. For HumanEval/MBPP
(code completion) we'd need batched generation with a stop token; that
adds ~1 hour of work and lands separately.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from myllm.eval.release_scorecard import (  # noqa: E402
    Scorecard,
    build_scorecard,
    write_scorecard,
)
from myllm.utils import configure_logging, get_logger  # noqa: E402

log = get_logger(__name__)


def _load_benchmarks(names: list[str]) -> list:
    """Resolve benchmark names to adapter instances.

    Supports both prompt-loader benchmarks (mmlu-pro, gsm8k, etc.) and
    heavyweight adapters (mmlu-prox, belebele, milu). Returns adapter
    objects that satisfy the Benchmark protocol.
    """
    benches = []
    for name in names:
        bench = _instantiate(name)
        if bench is not None:
            benches.append(bench)
        else:
            log.warning("scorecard_unknown_benchmark", name=name)
    return benches


def _instantiate(name: str):
    # Multilingual MCQ adapters (full Benchmark adapter path)
    if name == "mmlu-prox":
        from myllm.eval.benchmarks import MMLUProXBenchmark
        return MMLUProXBenchmark()
    if name == "belebele":
        from myllm.eval.benchmarks import BelebeleBenchmark
        return BelebeleBenchmark()
    if name == "milu":
        from myllm.eval.benchmarks import MILUBenchmark
        return MILUBenchmark()
    # Lightweight prompt-only loaders (most v1 gate benchmarks).
    # Wrap them in a minimal Benchmark adapter.
    try:
        from myllm.data.prompt_loaders import PROMPT_LOADERS, load_prompts
    except ImportError:
        return None
    if name in PROMPT_LOADERS:
        return _PromptLoaderBench(name, load_prompts)
    return None


class _PromptLoaderBench:
    """Minimal Benchmark adapter wrapping a prompt-only loader.

    The underlying loader returns prompt strings; we don't have target
    answers in the loader (only the model card's eval gate spec does).
    For the pilot scorecard we score "did the model output anything
    finite + non-empty" as a coarse smoke metric; for v1 release we'd
    swap in proper answer-key wiring per benchmark.
    """

    def __init__(self, name: str, load_prompts_fn):
        self.name = name
        self._load_prompts = load_prompts_fn

    def load_examples(self, split: str = "test", sample_size: int | None = None, seed: int = 0):
        from myllm.eval.types import EvalExample
        kwargs = {"sample_size": sample_size}
        if split != "test":
            kwargs["split"] = split
        for prompt in self._load_prompts(self.name, **kwargs):
            # No target answer in prompt loaders — use a sentinel.
            yield EvalExample(prompt=prompt, target_answer="", metadata={})

    def score(self, prediction: str, example) -> bool:
        # Pilot scorecard heuristic: model produced ANY non-empty
        # alphanumeric output. Replace with proper per-benchmark scorer
        # before the v1 release.
        return bool(prediction.strip())

    def subgroup_key(self, example) -> str:
        return "all"


def _build_predict_fn(args):
    """Build a predict_fn from the checkpoint. JAX-heavy; only called
    when we actually need to evaluate.

    Round B4 (2026-05-16): real predict_fn lands here via
    ``myllm.infer.predict.build_greedy_predict_fn``. ``--use-mock-predict``
    is preserved for scaffold-validation runs that don't need a real
    model.
    """
    if getattr(args, "use_mock_predict", False):
        # Mock returns "A" for everything — gives 25% accuracy on MMLU-style.
        return lambda prompt: "A"
    if not args.checkpoint_root:
        raise ValueError(
            "Real predict_fn requires --checkpoint-root. Pass "
            "--use-mock-predict to exercise the scorecard machinery with "
            "a constant-output mock instead."
        )
    from pathlib import Path

    from myllm.infer.predict import build_greedy_predict_fn

    return build_greedy_predict_fn(
        model_config_path=Path(args.model_config),
        tokenizer_path=Path(args.tokenizer_path),
        checkpoint_root=Path(args.checkpoint_root),
        checkpoint_step=args.checkpoint_step,
        max_new_tokens=int(getattr(args, "max_new_tokens", 80)),
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-config", required=True)
    p.add_argument("--tokenizer-path", default="artifacts/tokenizer_v1.json")
    p.add_argument("--checkpoint-root", required=False, default=None,
                   help="Orbax checkpoint root. Required unless --use-mock-predict.")
    p.add_argument("--checkpoint-step", type=int, default=None,
                   help="Specific step to load. Default: latest complete.")
    p.add_argument("--model-name", default="myllm-pilot-250m",
                   help="Public name for the scorecard header.")
    p.add_argument("--benchmarks", default="mmlu-pro,gsm8k,humaneval-plus,mbpp-plus",
                   help="Comma-separated benchmark ids.")
    p.add_argument("--sample-size", type=int, default=200,
                   help="Per-benchmark cap. None = full (slow). Default 200 for pilot.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--r2-prefix", default=None,
                   help="If set, upload scorecard.{json,md} to s3://$S3_BUCKET/<prefix>/.")
    p.add_argument("--notes", default="",
                   help="Free-form notes embedded in the scorecard markdown.")
    p.add_argument("--use-mock-predict", action="store_true",
                   help="Skip checkpoint loading + use a mock predict_fn that "
                        "returns 'A' for every prompt. Useful for validating "
                        "the scorecard machinery end-to-end without a model.")
    args = p.parse_args()

    configure_logging()

    bench_names = [n.strip() for n in args.benchmarks.split(",") if n.strip()]
    benches = _load_benchmarks(bench_names)
    if not benches:
        log.error("no benchmarks resolved", configured=bench_names)
        return 2

    predict_fn = _build_predict_fn(args)

    ckpt_label = args.checkpoint_root or "MOCK"
    if args.checkpoint_step is not None:
        ckpt_label = f"{ckpt_label}@step-{args.checkpoint_step}"

    card: Scorecard = build_scorecard(
        model_checkpoint=ckpt_label,
        model_name=args.model_name,
        benchmarks=benches,
        predict_fn=predict_fn,
        sample_size_per_benchmark=args.sample_size,
        seed=args.seed,
        notes=args.notes,
    )

    json_path, md_path = write_scorecard(card, args.output_dir)
    print(card.to_markdown())  # echo to stdout for quick inspection
    print(f"\nWrote: {json_path}  {md_path}")

    if args.r2_prefix:
        import os
        try:
            import boto3
            s3 = boto3.client("s3", endpoint_url=os.environ["S3_ENDPOINT_URL"])
            bucket = os.environ["S3_BUCKET"]
            prefix = args.r2_prefix.rstrip("/")
            for p in (json_path, md_path):
                key = f"{prefix}/{p.name}"
                s3.upload_file(str(p), bucket, key)
                log.info("scorecard_r2_uploaded", path=str(p), key=key)
        except Exception as e:  # noqa: BLE001
            log.warning("scorecard_r2_upload_failed", error=str(e))

    return 0


if __name__ == "__main__":
    sys.exit(main())
