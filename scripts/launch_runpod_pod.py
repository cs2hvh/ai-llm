#!/usr/bin/env python3
"""Launch / monitor / terminate RunPod pods. Skeleton — wired up in Phase 0 step 5.

The Phase 0 deliverable is just the smoke-test path: launch a tiny pod, run
`nvidia-smi`, tear down. Full training-launch logic lands in Phase 2.
"""
from __future__ import annotations

import argparse
import sys


def cmd_smoke(args: argparse.Namespace) -> int:
    print("[TODO] Phase 0 step 5: launch tiny pod, run nvidia-smi, tear down.")
    print(f"       requested SKU: {args.sku}")
    return 0


def cmd_launch(args: argparse.Namespace) -> int:
    print("[TODO] Phase 2+: launch training pod from config.")
    print(f"       config: {args.config}")
    return 0


def cmd_terminate(args: argparse.Namespace) -> int:
    print("[TODO] terminate pod by id.")
    print(f"       pod_id: {args.pod_id}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:  # noqa: ARG001
    print("[TODO] list active pods + cumulative spend.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="RunPod orchestration CLI (skeleton).")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("smoke", help="Phase 0 smoke test")
    sp.add_argument("--sku", default="1xA40", help="GPU SKU (default 1xA40; A10 was retired)")
    sp.set_defaults(func=cmd_smoke)

    sp = sub.add_parser("launch", help="Launch a training pod")
    sp.add_argument("--config", required=True, help="Path to YAML config")
    sp.set_defaults(func=cmd_launch)

    sp = sub.add_parser("terminate", help="Terminate a pod")
    sp.add_argument("--pod-id", required=True)
    sp.set_defaults(func=cmd_terminate)

    sp = sub.add_parser("list", help="List active pods")
    sp.set_defaults(func=cmd_list)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
