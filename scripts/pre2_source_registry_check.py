#!/usr/bin/env python3
"""Validate the pre-2 source registry and optional stage-readiness gate."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from myllm_pre2.data.source_registry import TrainingStage, load_source_registry


def _format_tokens(tokens: int) -> str:
    if tokens >= 1_000_000_000_000:
        return f"{tokens / 1_000_000_000_000:.2f}T"
    if tokens >= 1_000_000_000:
        return f"{tokens / 1_000_000_000:.1f}B"
    return f"{tokens:,}"


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        default=str(repo / "configs" / "data" / "pre2_source_registry.yaml"),
    )
    parser.add_argument(
        "--require-stage",
        choices=["canary", "poc", "proxy", "mainline"],
        default=None,
        help="Fail closed unless at least one approved source is allowed for this stage.",
    )
    parser.add_argument("--yaml", action="store_true", help="Emit machine-readable YAML.")
    args = parser.parse_args()

    registry = load_source_registry(args.registry)
    counts = registry.status_counts()
    stage_summary = {}
    for stage in ("canary", "poc", "proxy", "mainline"):
        sources = registry.sources_for_stage(stage)  # type: ignore[arg-type]
        stage_summary[stage] = {
            "approved_sources": len(sources),
            "estimated_tokens": registry.estimated_tokens_for_stage(stage),  # type: ignore[arg-type]
            "source_ids": [source.source_id for source in sources],
        }

    payload = {
        "name": registry.name,
        "status": registry.status,
        "sources": len(registry.sources),
        "license_status": counts,
        "buckets": registry.bucket_counts(),
        "stages": stage_summary,
    }

    if args.yaml:
        print(yaml.safe_dump(payload, sort_keys=False).strip())
    else:
        print(
            f"{registry.name}: sources={len(registry.sources)} "
            f"approved={counts['approved']} needs_review={counts['needs_review']} "
            f"blocked={counts['blocked']} excluded={counts['excluded']}"
        )
        for stage, summary in stage_summary.items():
            print(
                f"{stage}: approved_sources={summary['approved_sources']} "
                f"estimated_tokens={_format_tokens(summary['estimated_tokens'])}"
            )

    if args.require_stage is not None:
        required_stage = args.require_stage  # type: TrainingStage
        if not registry.sources_for_stage(required_stage):
            print(
                f"stage gate failed: no approved sources allowed for {required_stage}",
                file=sys.stderr,
            )
            return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
