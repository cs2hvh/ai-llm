#!/usr/bin/env python3
"""CLI entry point for the multi-agent research library.

Three subcommands matching the three workflows in
``src/myllm/research/workflows/``.

Usage:

    # Verify candidates from a YAML brief.
    python scripts/research_cli.py verify --brief briefs/teachers.yaml

    # Multi-source lookup.
    python scripts/research_cli.py lookup \\
        --question "What n-gram size do OLMo 2 / SmolLM3 / Llama 2 use for decon?" \\
        --source https://arxiv.org/abs/2402.16819 \\
        --source https://huggingface.co/blog/smollm3

    # Parallel file audit.
    python scripts/research_cli.py audit \\
        --audit "Find places that don't save data_position on resume" \\
        --path src/myllm/training/loop.py \\
        --path src/myllm/training/checkpoint.py

Outputs the synthesized answer to stdout and (optionally) writes
per-subagent reports + usage stats to a directory via --output-dir.

Briefs (for verify): YAML with shape::

    context: "Selecting a second teacher for MyLLM v1..."
    criteria:
      - "License must be permissive (Apache-2.0, MIT, or equivalent)"
      - "Must be a base text-only model"
    candidates:
      - id: olmo-3-32b
        hf_id: allenai/Olmo-3-1125-32B
        notes: 32B dense, 5.5T tokens
      - id: qwen3-14b
        hf_id: Qwen/Qwen3-14B-Base
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

import yaml  # noqa: E402

from myllm.research import (  # noqa: E402
    ResearchClient,
    multi_source_lookup,
    parallel_audit,
    verify_candidates,
)
from myllm.research.workflows.base import WorkflowResult  # noqa: E402
from myllm.utils import configure_logging, get_logger  # noqa: E402

log = get_logger(__name__)


def _write_output(result: WorkflowResult, output_dir: Path | None) -> None:
    """Print to stdout + optionally persist per-subagent + usage to disk."""
    print(result.summary)
    print()
    print(f"--- usage: {result.usage.summary()} ---")
    if output_dir is None:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.md").write_text(result.summary)
    (output_dir / "usage.json").write_text(
        json.dumps(
            {
                "input_tokens": result.usage.input_tokens,
                "output_tokens": result.usage.output_tokens,
                "n_calls": result.usage.n_calls,
                "n_cache_hits": result.usage.n_cache_hits,
                "usd_cost": result.usage.usd_cost,
            },
            indent=2,
        )
    )
    subagents_dir = output_dir / "subagents"
    subagents_dir.mkdir(exist_ok=True)
    for sid, r in result.subagent_outputs.items():
        safe = sid.replace("/", "__")
        (subagents_dir / f"{safe}.md").write_text(
            r.content if not r.error else f"FAILED: {r.error}\n"
        )
    log.info("research_output_written", dir=str(output_dir))


def _cmd_verify(args, client: ResearchClient) -> int:
    brief = yaml.safe_load(Path(args.brief).read_text())
    if not isinstance(brief, dict):
        print(f"ERROR: {args.brief} is not a YAML mapping", file=sys.stderr)
        return 2
    candidates = brief.get("candidates") or []
    criteria = brief.get("criteria") or []
    context = brief.get("context", "")
    if not candidates or not criteria:
        print(
            f"ERROR: brief must have 'candidates' and 'criteria' keys; "
            f"got keys: {list(brief)}",
            file=sys.stderr,
        )
        return 2
    result = verify_candidates(
        candidates=candidates,
        criteria=criteria,
        context=context,
        client=client,
        max_workers=args.max_workers,
    )
    _write_output(result, Path(args.output_dir) if args.output_dir else None)
    return 0


def _cmd_lookup(args, client: ResearchClient) -> int:
    if not args.source:
        print("ERROR: at least one --source required", file=sys.stderr)
        return 2
    result = multi_source_lookup(
        question=args.question,
        sources=args.source,
        client=client,
        max_workers=args.max_workers,
    )
    _write_output(result, Path(args.output_dir) if args.output_dir else None)
    return 0


def _cmd_audit(args, client: ResearchClient) -> int:
    if not args.path:
        print("ERROR: at least one --path required", file=sys.stderr)
        return 2
    result = parallel_audit(
        paths=args.path,
        audit_question=args.audit,
        repo_root=args.repo_root,
        client=client,
        max_workers=args.max_workers,
    )
    _write_output(result, Path(args.output_dir) if args.output_dir else None)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    # Common args go on the top-level parser so subcommands inherit them.
    p.add_argument(
        "--model",
        default="claude-opus-4-7",
        help="Model id (default: claude-opus-4-7).",
    )
    p.add_argument(
        "--cache-dir",
        default=str(_REPO / "artifacts" / "research_cache"),
        help="Disk cache directory for completed runs (default: artifacts/research_cache).",
    )
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable the disk cache (always re-run subagents).",
    )
    p.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="Max concurrent subagents (default 5).",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Persist summary + per-subagent reports + usage.json to this dir.",
    )

    sub = p.add_subparsers(dest="cmd", required=True)

    # verify
    sp = sub.add_parser("verify", help="Verify N candidates against M criteria.")
    sp.add_argument("--brief", required=True, help="YAML brief (see module docstring).")

    # lookup
    sp = sub.add_parser("lookup", help="Answer a question by reading N sources in parallel.")
    sp.add_argument("--question", required=True)
    sp.add_argument(
        "--source", action="append", default=[],
        help="URL of a source (pass repeatedly for multiple).",
    )

    # audit
    sp = sub.add_parser("audit", help="Audit N files for a single concern in parallel.")
    sp.add_argument("--audit", required=True, help="The audit question.")
    sp.add_argument(
        "--path", action="append", default=[],
        help="Repo-relative file path (pass repeatedly for multiple).",
    )
    sp.add_argument(
        "--repo-root", default=str(_REPO),
        help="Repo root (default: this repo's root).",
    )

    args = p.parse_args()

    configure_logging()
    client = ResearchClient(
        model=args.model,
        cache_dir=None if args.no_cache else args.cache_dir,
    )

    if args.cmd == "verify":
        return _cmd_verify(args, client)
    if args.cmd == "lookup":
        return _cmd_lookup(args, client)
    if args.cmd == "audit":
        return _cmd_audit(args, client)
    print(f"unknown command: {args.cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
