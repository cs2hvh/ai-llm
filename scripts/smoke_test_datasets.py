#!/usr/bin/env python3
"""Preflight: smoke-test every data source before training pod launch.

The 2026-05-11 dossier audit added this as the Section 6 preflight check.
Without it, dataset-side bugs (gating, trust_remote_code, missing config_name,
missing split, decompression codec, fragile loader scripts) only surface
mid-cell on the training pod — burning $$.

This script pulls N rows from each source listed in a data yaml and prints
green/red status. Run on the control plane (no GPU needed) BEFORE booking
a training pod.

Usage:
    # Check the production mix:
    python scripts/smoke_test_datasets.py --data-config configs/data/pretrain_mix.yaml

    # Check more aggressively (e.g. 50 rows to surface mid-stream errors):
    python scripts/smoke_test_datasets.py --data-config configs/data/pretrain_mix.yaml --sample-size 50

Exit codes:
    0 — all sources passed
    1 — one or more sources failed (paste output to triage)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

import yaml  # noqa: E402

from myllm.data.loader import HFStreamLoader  # noqa: E402


def _check_source(entry: dict, *, sample_size: int) -> tuple[bool, str, float]:
    """Return (passed, message, elapsed_seconds)."""
    t0 = time.time()
    try:
        loader = HFStreamLoader(
            dataset=entry["dataset"],
            category=entry["category"],
            text_field=entry.get("text_field", "text"),
            config_name=entry.get("config_name"),
            split=entry.get("split", "train"),
            trust_remote_code=bool(entry.get("trust_remote_code", False)),
            sample_limit=sample_size,
        )
        seen = 0
        total_chars = 0
        for doc in loader:
            seen += 1
            total_chars += len(doc.text)
            if seen >= sample_size:
                break
        elapsed = time.time() - t0
        if seen == 0:
            return False, "iterator yielded zero rows", elapsed
        return True, f"{seen} rows, avg_len={total_chars // seen} chars", elapsed
    except Exception as e:
        elapsed = time.time() - t0
        # Take the most actionable line of the exception
        trace = traceback.format_exception_only(type(e), e)[-1].strip()
        return False, trace, elapsed


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-config", required=True)
    p.add_argument("--sample-size", type=int, default=5,
                   help="Rows to pull per source (default 5).")
    p.add_argument("--require-hf-token", action="store_true",
                   help="Fail fast if HF_TOKEN isn't set.")
    args = p.parse_args()

    if args.require_hf_token and not os.environ.get("HF_TOKEN"):
        print("ERROR: HF_TOKEN not exported; gated datasets will fail.")
        return 1

    with open(args.data_config) as f:
        cfg = yaml.safe_load(f)

    sources = cfg.get("sources", [])
    print(f"\nsmoke-testing {len(sources)} sources from {args.data_config}")
    print(f"sample_size = {args.sample_size} rows per source\n")
    print(f"{'#':<3} {'dataset':<48} {'config':<14} {'split':<8} {'time':<8} status")
    print("-" * 130)

    passed = 0
    failed = 0
    failures: list[str] = []
    for i, entry in enumerate(sources, 1):
        ok, msg, elapsed = _check_source(entry, sample_size=args.sample_size)
        status = "PASS" if ok else "FAIL"
        symbol = "OK" if ok else "XX"
        print(
            f"{i:<3} {entry['dataset']:<48} "
            f"{(entry.get('config_name') or '-'):<14} "
            f"{(entry.get('split') or 'train'):<8} "
            f"{elapsed:>5.1f}s   "
            f"[{symbol}] {msg[:60]}"
        )
        if ok:
            passed += 1
        else:
            failed += 1
            failures.append(f"  {entry['dataset']} ({entry.get('config_name', '-')}): {msg}")

    print()
    print(f"summary: {passed}/{len(sources)} passed, {failed} failed")
    if failures:
        print("\nfailures:")
        for f in failures:
            print(f)
        return 1
    print("\nAll sources accessible. Safe to launch training pod.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
