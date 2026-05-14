#!/usr/bin/env python3
"""Validate pre-2 planning configs.

This is the first executable pre-2 contract. It validates model/data configs
without importing TorchTitan or touching the pre-1 training stack.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from myllm_pre2.config import load_data_mix_config, load_dense_config


def _format_tokens(n: int) -> str:
    if n >= 1_000_000_000_000:
        return f"{n / 1_000_000_000_000:.2f}T"
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B"
    return f"{n:,}"


def _format_params(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.3f}B"
    return f"{n / 1_000_000:.1f}M"


def _format_flops(n: int) -> str:
    return f"{n:.3e}"


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-config",
        action="append",
        default=None,
        help="Pre-2 dense model config to validate.",
    )
    parser.add_argument(
        "--include-poc-ladder",
        action="store_true",
        help="Validate the canary, POC, proxy, and mainline dense configs.",
    )
    parser.add_argument(
        "--data-config",
        action="append",
        default=None,
        help="Pre-2 data mix config to validate. Can be repeated.",
    )
    args = parser.parse_args()
    if args.model_config is not None:
        model_configs = args.model_config
    elif args.include_poc_ladder:
        model_configs = [
            str(repo / "configs" / "pre2_dense_canary_110m.yaml"),
            str(repo / "configs" / "pre2_dense_poc_250m.yaml"),
            str(repo / "configs" / "pre2_dense_proxy_400m.yaml"),
            str(repo / "configs" / "pre2_dense_1_5b.yaml"),
        ]
    else:
        model_configs = [str(repo / "configs" / "pre2_dense_1_5b.yaml")]
    data_configs = args.data_config or [
        str(repo / "configs" / "data" / "pre2_mix_stage1.yaml"),
        str(repo / "configs" / "data" / "pre2_mix_stage2.yaml"),
        str(repo / "configs" / "data" / "pre2_mix_anneal.yaml"),
    ]

    mainline_cfg = None
    for model_path in model_configs:
        model_cfg = load_dense_config(model_path)
        if model_cfg.name.endswith("1.5b-base"):
            mainline_cfg = model_cfg
        estimate = model_cfg.parameter_count_estimate()
        budget = model_cfg.training.token_budget
        print(
            f"{model_cfg.name}: estimated_params={_format_params(estimate)} "
            f"target_delta={model_cfg.parameter_target_delta():.2%}"
        )
        if budget.study_target is not None:
            steps = model_cfg.training_steps_for_tokens(budget.study_target)
            target_steps = f"{steps:,}" if steps is not None else "n/a"
            batch_tokens = (
                f"{model_cfg.training.batch.global_batch_tokens:,}"
                if model_cfg.training.batch and model_cfg.training.batch.global_batch_tokens
                else "n/a"
            )
            print(
                "study_decision: "
                f"minimum={_format_tokens(budget.study_minimum) if budget.study_minimum else 'n/a'} "
                f"target={_format_tokens(budget.study_target)} "
                f"stretch={_format_tokens(budget.study_stretch) if budget.study_stretch else 'n/a'} "
                f"context={model_cfg.model.context.foundation_length:,} "
                f"global_batch_tokens={batch_tokens} "
                f"target_steps={target_steps}"
            )
        if budget.internal_minimum is not None and budget.release_target is not None:
            print(
                "decision: "
                f"internal_minimum={_format_tokens(budget.internal_minimum)} "
                f"release_target={_format_tokens(budget.release_target)} "
                f"stretch={_format_tokens(budget.stretch) if budget.stretch else 'n/a'} "
                f"context={model_cfg.model.context.foundation_length:,} "
                f"global_batch_tokens={model_cfg.training.batch.global_batch_tokens:,}"
            )
            release_steps = model_cfg.training_steps_for_tokens(budget.release_target)
            if release_steps is not None:
                print(
                    "release_compute: "
                    f"steps={release_steps:,} "
                    f"dense_flops_6NT={_format_flops(model_cfg.dense_training_flops(budget.release_target))}"
                )

    total_data_tokens = 0
    for path in data_configs:
        data_cfg = load_data_mix_config(path)
        total_data_tokens += data_cfg.target_tokens
        print(
            f"{data_cfg.name}: target_tokens={data_cfg.target_tokens:,} "
            f"buckets={len(data_cfg.source_buckets)} synthetic_cap={data_cfg.synthetic_cap:.0%}"
        )
    if mainline_cfg is not None and mainline_cfg.training.token_budget.release_target is not None:
        release_target = mainline_cfg.training.token_budget.release_target
        print(
            "data_plan: "
            f"configured_stage_tokens={_format_tokens(total_data_tokens)} "
            f"release_target_match={total_data_tokens == release_target}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
