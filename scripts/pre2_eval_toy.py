#!/usr/bin/env python3
"""Run a tiny real next-token eval against a pre-2 smoke checkpoint."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import yaml

from myllm_pre2.config import load_dense_config
from myllm_pre2.eval import NextTokenEvalExample, run_checkpoint_next_token_eval


def _parse_example(raw: str) -> NextTokenEvalExample:
    try:
        prompt_raw, target_raw = raw.split(":", 1)
        prompt_ids = [int(part.strip()) for part in prompt_raw.split(",") if part.strip()]
        target_id = int(target_raw.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "examples must use comma-separated prompt ids followed by ':target', "
            "for example '1,2,3:4'"
        ) from exc
    return NextTokenEvalExample(prompt_ids=prompt_ids, target_id=target_id)


def _result_to_dict(result) -> dict:
    return {
        "checkpoint_step": result.checkpoint_step,
        "num_examples": result.num_examples,
        "accuracy": result.accuracy,
        "average_nll": result.average_nll,
        "predictions": [
            {
                "prompt_ids": prediction.prompt_ids,
                "target_id": prediction.target_id,
                "predicted_id": prediction.predicted_id,
                "correct": prediction.correct,
                "nll": prediction.nll,
            }
            for prediction in result.predictions
        ],
    }


def _make_tiny_smoke_config(repo: Path, base_cfg):
    script = repo / "scripts" / "pre2_train.py"
    spec = importlib.util.spec_from_file_location("pre2_train_for_eval", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.make_tiny_smoke_config(base_cfg)


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-config",
        default=str(repo / "configs" / "pre2_dense_1_5b.yaml"),
        help="Base pre-2 model config. Uses a tiny derived config unless --full-config is set.",
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--full-config",
        action="store_true",
        help="Use the supplied config directly instead of the tiny smoke derivative.",
    )
    parser.add_argument(
        "--example",
        action="append",
        type=_parse_example,
        default=None,
        help="Toy next-token example as 'prompt_id,prompt_id:target_id'. Can be repeated.",
    )
    args = parser.parse_args()

    base_cfg = load_dense_config(args.model_config)
    if args.full_config:
        cfg = base_cfg
    else:
        cfg = _make_tiny_smoke_config(repo, base_cfg)

    examples = args.example or [
        NextTokenEvalExample(prompt_ids=[1, 2, 3], target_id=4),
        NextTokenEvalExample(prompt_ids=[5, 6, 7], target_id=8),
    ]
    result = run_checkpoint_next_token_eval(
        cfg,
        args.checkpoint_dir,
        examples,
        device=args.device,
    )
    print(yaml.safe_dump(_result_to_dict(result), sort_keys=False).strip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
