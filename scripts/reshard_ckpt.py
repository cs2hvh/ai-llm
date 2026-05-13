#!/usr/bin/env python3
"""Reshard an Orbax checkpoint onto a different mesh layout.

Use case: development saves a checkpoint on 1 device (or pre-FSDP DP-
replicated state); production wants to resume on 5x H200 with FSDP. Or
vice versa — load a sharded checkpoint onto a single GPU for eval.

Usage:
    python scripts/reshard_ckpt.py \\
        --src /path/to/ckpt-dir \\
        --dst /path/to/new-ckpt-dir \\
        --src-step 5000 \\
        --target-devices 5

Memory: at 1B params (~20 GB state) the intermediate live copy on the
host's default device is fine. For 7B+ models the right path is an
Orbax template-based reshard (not implemented here; this is the v1 1B
workflow's actual need).

FSDP Commit G (2026-05-13). Together with Commits A-F, this closes the
"checkpoint portability under sharded state" item the senior reviewer
flagged in the FSDP gauntlet.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Keras + JAX backend setup before any model imports.
os.environ.setdefault("KERAS_BACKEND", "jax")

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from myllm.training.checkpoint import reshard_checkpoint  # noqa: E402
from myllm.utils import configure_logging, get_logger  # noqa: E402

log = get_logger(__name__)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--src", required=True,
        help="Source checkpoint root directory.",
    )
    p.add_argument(
        "--dst", required=True,
        help="Destination root directory. Created if missing. The reshard "
             "step dir is written at dst/step-NNNNNNNNN/.",
    )
    p.add_argument(
        "--src-step", type=int, required=True,
        help="Source step number to load. Use scripts/find_resume_step.py "
             "if unsure (or read the manifests directly).",
    )
    p.add_argument(
        "--target-devices", type=int, required=True,
        help="Number of data-parallel devices in the target mesh. The "
             "destination state will be sharded along the data axis "
             "across these devices.",
    )
    p.add_argument(
        "--target-mesh-axis", default="data",
        help="Name of the data axis in the target mesh (default: 'data', "
             "matches build_mesh_and_shardings).",
    )
    args = p.parse_args()

    configure_logging()

    out_path = reshard_checkpoint(
        src_root=args.src,
        dst_root=args.dst,
        src_step=args.src_step,
        target_devices=args.target_devices,
        target_mesh_axis=args.target_mesh_axis,
    )
    log.info(
        "reshard_complete",
        src=args.src,
        dst=args.dst,
        src_step=args.src_step,
        target_devices=args.target_devices,
        output_path=str(out_path),
    )
    print(f"resharded checkpoint written to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
