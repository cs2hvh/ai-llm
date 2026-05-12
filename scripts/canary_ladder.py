#!/usr/bin/env python3
"""Canary ladder CLI — pre-launch validation gates.

Per 2026-05-12 reviewer Q&A §5 and plan_v3 §4: "the process that saves
the project is a canary ladder with hard acceptance gates." This runner
exercises the CPU-runnable stages (L0, L5) and emits structured pass/fail
records.

Stages that need a GPU pod (L1, L2, L4) are documented but skipped here —
their pass criteria + runbook are in the print output. Run them from a
GPU pod via the templates printed.

L3 (forced-kill resume bitwise-exact) is its own script
(``scripts/canary_l3_resume.py``) because it spawns subprocesses; run
it separately or invoke via ``--include-l3``.

Usage::

    # CPU-only stages on the base 1B config + a corpus root:
    python scripts/canary_ladder.py \\
        --model-config configs/base_1b.yaml \\
        --tokenizer-path artifacts/tokenizer_v1.json \\
        --packed-corpus-root /data/v1/train

    # Same, plus L3 in-process (slower; ~30s on CPU):
    python scripts/canary_ladder.py ... --include-l3

    # JSON output for CI parsing:
    python scripts/canary_ladder.py ... --format json

Exit code: 0 if all stages pass, 1 otherwise. Suitable for a CI gate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from myllm.canary import (  # noqa: E402
    StageResult,
    report_to_json,
    report_to_text,
    run_l0,
    run_l5,
)
from myllm.utils import configure_logging  # noqa: E402


_GPU_STAGES_RUNBOOK = """\
The remaining canary stages require GPU pod time. Run from the pod after
SSH-ing in, with the same config as the production base run:

  L1 — Single-GPU 20-step smoke (~2 min, ~$0.20):
      python scripts/run_pretrain.py \\
          --model-config {model_config} \\
          --tokenizer-path {tokenizer_path} \\
          --synthetic-data \\
          --total-steps 20 \\
          --checkpoint-every 100   # don't checkpoint mid-smoke
      # Pass criteria:
      #   - CE loss strictly decreasing across the 20 steps
      #   - No NCCL/kernel errors

  L2 — Multi-GPU 8x parallelism check (~2 min, ~$1):
      # Run L1 with --no-shard removed (multi-device is default).
      # Capture the single-GPU loss from L1; multi-GPU loss should match
      # within 1e-6 (fp32 master copy).

  L4 — 1B-shape 1-2% scale rehearsal (~1 hr, ~$30-100):
      python scripts/run_pretrain.py \\
          --model-config configs/base_1b.yaml \\
          --tokenizer-path {tokenizer_path} \\
          --packed-corpus-root <REAL_CORPUS> \\
          --total-steps 1500   # ~1% of a 1T run at micro_batch=8 / seq=8192
      # Pass criteria:
      #   - Sustained >=35% MFU over 1 hour
      #   - No NaN events (atomic NaN-skip should fire 0 times on canary data)
      #   - Per-source val curves all monotone in the stable phase
"""


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-config", required=True,
                   help="Path to model yaml (configs/base_1b.yaml etc.).")
    p.add_argument("--tokenizer-path", required=True,
                   help="Path to tokenizer.json.")
    p.add_argument("--packed-corpus-root", default=None,
                   help="If set, run L5 packed-corpus data sanity checks on this corpus.")
    p.add_argument("--expected-params", type=int, default=None,
                   help="If set, L0 verifies the model's actual param count against this.")
    p.add_argument("--source-share-tolerance", type=float, default=0.02,
                   help="Allowed |actual - target| per source (default 2%%).")
    p.add_argument("--include-l3", action="store_true",
                   help="Also run L3 forced-kill resume test in-process.")
    p.add_argument("--format", choices=("text", "json"), default="text",
                   help="Output format.")
    p.add_argument("--print-gpu-runbook", action="store_true",
                   help="Print the L1/L2/L4 GPU-pod runbook + exit.")
    args = p.parse_args()

    configure_logging()

    if args.print_gpu_runbook:
        print(_GPU_STAGES_RUNBOOK.format(
            model_config=args.model_config, tokenizer_path=args.tokenizer_path,
        ))
        return 0

    # Read model config for vocab_size (L5 needs it).
    model_cfg = yaml.safe_load(Path(args.model_config).read_text())
    vocab_size = int(model_cfg["vocab_size"])

    stages: list[StageResult] = []

    # L0 — static checks (always run).
    stages.append(run_l0(
        model_config_path=args.model_config,
        tokenizer_path=args.tokenizer_path,
        expected_params=args.expected_params,
    ))

    # L5 — packed corpus sanity (only if a corpus is provided).
    if args.packed_corpus_root is not None:
        stages.append(run_l5(
            corpus_root=args.packed_corpus_root,
            vocab_size=vocab_size,
            source_share_tolerance=args.source_share_tolerance,
        ))

    # L3 — optional, in-process.
    if args.include_l3:
        from myllm.canary import StageResult, CheckResult
        try:
            from scripts.canary_l3_resume import run_l3_check
            l3 = run_l3_check()
            stages.append(StageResult(stage="L3", checks=[l3]))
        except ModuleNotFoundError as e:
            stages.append(StageResult(
                stage="L3",
                checks=[CheckResult(
                    name="l3_forced_kill_resume",
                    passed=False,
                    summary=f"import failed: {e}",
                    fix_hint="Ensure keras + jax installed; KERAS_BACKEND=jax.",
                )],
            ))

    # Output.
    if args.format == "json":
        print(report_to_json(stages))
    else:
        print(report_to_text(stages))
        if any(s.stage == "L4" or s.stage == "L1" or s.stage == "L2"
               for s in stages):
            pass  # GPU stages were run
        else:
            print()
            print("(L1/L2/L4 require GPU pod — run scripts/canary_ladder.py "
                  "--print-gpu-runbook for the commands.)")

    overall = all(s.passed for s in stages)
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
