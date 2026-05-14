#!/usr/bin/env python3
"""Emit per-source packed-corpus build commands for the pre-2 v0.5 POC."""
from __future__ import annotations

import argparse
import shlex
from pathlib import Path
from typing import Any

import yaml


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not parse to a mapping")
    return data


def _ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _format_command(args: list[str], shell: str) -> str:
    if shell == "powershell":
        return " ".join(_ps_quote(arg) if any(c.isspace() for c in arg) else arg for arg in args)
    return " ".join(shlex.quote(arg) for arg in args)


def build_commands(
    *,
    repo: Path,
    build_mix_path: Path,
    output_root: str,
    tokenizer_path: str,
    sequence_length: int,
    sequences_per_shard: int,
    production: bool,
    shell: str,
) -> list[str]:
    build_mix = _load_yaml(build_mix_path)
    commands: list[str] = []
    for source in build_mix.get("sources", []):
        source_id = str(source["source_id"])
        revision = str(source["revision"])
        target_tokens = str(int(source["target_tokens"]))
        args = [
            "python",
            "scripts/build_packed_corpus.py",
            "--pretrain-mix-config",
            str(build_mix_path),
            "--source",
            source_id,
            "--source-id",
            source_id,
            "--tokenizer-path",
            tokenizer_path,
            "--output-root",
            output_root,
            "--sequence-length",
            str(sequence_length),
            "--sequences-per-shard",
            str(sequences_per_shard),
            "--revision-id",
            revision,
            "--hf-revision",
            revision,
            "--target-tokens",
            target_tokens,
        ]
        if production:
            args.append("--production")
        commands.append(_format_command(args, shell))
    return commands


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build-mix",
        default=str(repo / "configs" / "data" / "pre2_v0_5_build_mix.yaml"),
    )
    parser.add_argument("--output-root", default="/data/pre2/v0_5/sources")
    parser.add_argument("--tokenizer-path", default="artifacts/tokenizer_v1.json")
    parser.add_argument("--sequence-length", type=int, default=8193)
    parser.add_argument("--sequences-per-shard", type=int, default=65536)
    parser.add_argument("--no-production", action="store_true")
    parser.add_argument("--shell", choices=["posix", "powershell"], default="posix")
    args = parser.parse_args()

    commands = build_commands(
        repo=repo,
        build_mix_path=Path(args.build_mix),
        output_root=args.output_root,
        tokenizer_path=args.tokenizer_path,
        sequence_length=args.sequence_length,
        sequences_per_shard=args.sequences_per_shard,
        production=not args.no_production,
        shell=args.shell,
    )
    for command in commands:
        print(command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
