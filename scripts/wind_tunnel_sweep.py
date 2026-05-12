#!/usr/bin/env python3
"""muP wind-tunnel hyperparameter sweep — R1 follow-up.

Runs a grid of (peak_lr, init_std) cells against the 30M proxy model in
``configs/wind_tunnel.yaml``. Each cell trains for ``--tokens-per-cell``
tokens, the final training loss is captured, and a manifest of results
is written to disk. The optimal cell's hyperparameters transfer zero-shot
to pilot 250M and base 1B under muP (per ``docs/mup_design.md``).

Cost reference (30M proxy, 200M tokens/cell, 1× B200): ~$3-5 per cell,
~$30-50 for the full 10-cell grid. Wall time: ~1 hr per cell at ~50K
tokens/sec effective throughput (single device, micro_batch=8, seq=2048
→ ~16K tok/step → ~12K steps). NOTE: an earlier version of this script
defaulted to 1B tokens/cell, which would cost ~$200; we dropped to 200M
based on μP literature (HP signal is clear at 100-300M for a 30M model).

Usage:

    # Dry-run (default) — print what would be executed, no compute.
    python scripts/wind_tunnel_sweep.py

    # Generate per-cell launch commands without running them.
    python scripts/wind_tunnel_sweep.py --output artifacts/wind_tunnel_plan.json

    # Execute the sweep locally (each cell as a subprocess of run_pretrain.py).
    python scripts/wind_tunnel_sweep.py --execute --output artifacts/wind_tunnel_results.json

    # After all cells done, collect results and report the optimum.
    python scripts/wind_tunnel_sweep.py --collect --output artifacts/wind_tunnel_results.json

Grid: peak_lr ∈ {5e-4, 1e-3, 2e-3, 4e-3, 8e-3} × init_std ∈ {0.01, 0.02}.
That's 10 cells. We sweep peak_lr more densely because LR sensitivity is
larger than init sensitivity at our scale.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Grid definition
# --------------------------------------------------------------------------- #
DEFAULT_LR_GRID = (5.0e-4, 1.0e-3, 2.0e-3, 4.0e-3, 8.0e-3)
DEFAULT_INIT_GRID = (0.01, 0.02)


@dataclass
class SweepCell:
    """One (peak_lr, init_std) cell of the sweep."""

    cell_id: str
    peak_lr: float
    init_std: float
    tokens: int
    final_loss: float | None = None
    elapsed_seconds: float | None = None
    run_log_path: str | None = None
    status: str = "pending"  # pending | running | done | failed
    error: str | None = None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class SweepManifest:
    """Top-level sweep manifest written to disk."""

    config_path: str
    data_config_path: str
    tokenizer_path: str
    tokens_per_cell: int
    cells: list[SweepCell] = field(default_factory=list)
    started_at: str | None = None
    completed_at: str | None = None

    def to_dict(self) -> dict:
        return {
            "config_path": self.config_path,
            "data_config_path": self.data_config_path,
            "tokenizer_path": self.tokenizer_path,
            "tokens_per_cell": self.tokens_per_cell,
            "cells": [c.to_dict() for c in self.cells],
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


def make_cell_id(lr: float, init: float) -> str:
    """Stable filename-safe ID like ``lr1e-03_init2e-02``."""
    return f"lr{lr:.0e}_init{init:.0e}".replace("+0", "").replace("-0", "-")


def build_grid(
    lr_grid: tuple[float, ...] = DEFAULT_LR_GRID,
    init_grid: tuple[float, ...] = DEFAULT_INIT_GRID,
    tokens_per_cell: int = 200_000_000,
) -> list[SweepCell]:
    """Return a list of SweepCells covering the full grid."""
    cells = []
    for lr in lr_grid:
        for init in init_grid:
            cells.append(
                SweepCell(
                    cell_id=make_cell_id(lr, init),
                    peak_lr=lr,
                    init_std=init,
                    tokens=tokens_per_cell,
                )
            )
    return cells


# --------------------------------------------------------------------------- #
# Cell launch helpers
# --------------------------------------------------------------------------- #
def _read_model_yaml(model_config_path: str) -> dict:
    import yaml as _yaml
    with open(model_config_path) as f:
        return _yaml.safe_load(f)


def _read_data_yaml(data_config_path: str) -> dict:
    """Read the data yaml; return {} if path is a stub (test convenience)."""
    import yaml as _yaml
    try:
        with open(data_config_path) as f:
            return _yaml.safe_load(f) or {}
    except (FileNotFoundError, OSError):
        return {}


def _read_context_length(model_config_path: str) -> int:
    """Read the proxy model's context_length from its yaml. Authoritative
    source for seq-len math — see P0-3 fix in run_pretrain.py."""
    cfg = _read_model_yaml(model_config_path)
    if "context_length" not in cfg:
        raise ValueError(
            f"{model_config_path}: missing required 'context_length' field"
        )
    return int(cfg["context_length"])


def _resolve_micro_batch_from_yamls(
    model_yaml: dict | None,
    data_yaml: dict | None,
    default: int = 8,
) -> int:
    """Pure helper mirroring run_pretrain.resolve_micro_batch's priority
    order (model > data > default). The sweep can't accept a CLI override
    on each cell — that's set per-cell via the cell config — so this
    helper omits the CLI tier.
    """
    if model_yaml and "batch" in model_yaml and "micro_batch_per_device" in model_yaml["batch"]:
        return int(model_yaml["batch"]["micro_batch_per_device"])
    if data_yaml and "batch" in data_yaml and "micro_batch_per_device" in data_yaml["batch"]:
        return int(data_yaml["batch"]["micro_batch_per_device"])
    return int(default)


def cell_command(
    cell: SweepCell,
    model_config: str,
    data_config: str,
    tokenizer_path: str,
    checkpoint_root: str,
    log_path: str,
    micro_batch_per_device: int | None = None,
    sequence_length: int | None = None,
) -> list[str]:
    """Return the argv list for running one cell via scripts/run_pretrain.py.

    `sequence_length` defaults to the model config's `context_length` (the
    authoritative source post 2026-05-12 audit). Passing it explicitly is
    allowed but must match; mismatch is detected at run_pretrain.py startup.

    `micro_batch_per_device` follows the same resolver as run_pretrain.py:
        model yaml's batch.micro_batch_per_device > data yaml's >
        hardcoded 8. Explicit kwarg here is highest priority (test convenience).
    Re-audit 2026-05-12: previously hardcoded to 8 here, which silently
    ignored Proxy B's micro_batch=4 setting and would have caused OOM at
    sweep time.
    """
    if sequence_length is None:
        sequence_length = _read_context_length(model_config)
    if micro_batch_per_device is None:
        model_yaml = _read_model_yaml(model_config) if model_config else {}
        data_yaml = _read_data_yaml(data_config) if data_config else {}
        micro_batch_per_device = _resolve_micro_batch_from_yamls(
            model_yaml, data_yaml, default=8
        )

    # tokens per step = micro_batch × seq_len × devices (we assume 1 device for
    # the sweep — 30M model fits comfortably on a single H100/B200).
    tokens_per_step = micro_batch_per_device * sequence_length
    total_steps = max(1, math.ceil(cell.tokens / tokens_per_step))
    return [
        sys.executable, "-u",
        str(_REPO / "scripts" / "run_pretrain.py"),
        "--model-config", model_config,
        "--data-config", data_config,
        "--tokenizer-path", tokenizer_path,
        "--run-name", f"wind_tunnel_{cell.cell_id}",
        "--total-steps", str(total_steps),
        "--checkpoint-root", checkpoint_root,
        "--checkpoint-every", "10000",  # don't checkpoint for sweep cells
        "--log-every", "100",
        "--no-wandb",
        "--no-watchdog",  # cells must run to completion; spike → high loss IS the signal we want
        "--peak-lr-override", repr(cell.peak_lr),
        "--init-std-override", repr(cell.init_std),
        # Pass micro_batch explicitly so the launched run_pretrain agrees
        # with this script's tokens_per_step math (re-audit 2026-05-12 fix).
        "--micro-batch-override", str(micro_batch_per_device),
    ]


_LOSS_RE = re.compile(r'"loss"\s*:\s*([0-9eE.+-]+)')


def parse_final_loss(log_path: Path) -> float | None:
    """Extract the last ``"loss": ...`` value from a training log."""
    if not log_path.exists():
        return None
    last_loss = None
    with open(log_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = _LOSS_RE.search(line)
            if m:
                try:
                    last_loss = float(m.group(1))
                except ValueError:
                    continue
    return last_loss


def run_cell(
    cell: SweepCell,
    model_config: str,
    data_config: str,
    tokenizer_path: str,
    artifact_root: Path,
) -> SweepCell:
    """Execute one sweep cell. Mutates and returns the cell with results."""
    log_dir = artifact_root / "wind_tunnel" / cell.cell_id
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train.log"
    ckpt_root = str(log_dir / "ckpts")

    cmd = cell_command(
        cell,
        model_config=model_config,
        data_config=data_config,
        tokenizer_path=tokenizer_path,
        checkpoint_root=ckpt_root,
        log_path=str(log_path),
    )
    cell.status = "running"
    cell.run_log_path = str(log_path)
    t0 = time.time()
    try:
        with open(log_path, "w", encoding="utf-8") as logf:
            res = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
        cell.elapsed_seconds = round(time.time() - t0, 1)
        if res.returncode != 0:
            cell.status = "failed"
            cell.error = f"run_pretrain.py exited {res.returncode}"
        else:
            cell.final_loss = parse_final_loss(log_path)
            cell.status = "done" if cell.final_loss is not None else "failed"
            if cell.final_loss is None:
                cell.error = "no loss values found in log"
    except Exception as e:
        cell.elapsed_seconds = round(time.time() - t0, 1)
        cell.status = "failed"
        cell.error = f"{type(e).__name__}: {e}"
    return cell


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def report_grid(cells: list[SweepCell]) -> str:
    """Format the sweep result table."""
    lines = [
        f"{'cell_id':<18} | {'peak_lr':>10} | {'init_std':>9} | {'tokens':>14} | {'final_loss':>11} | {'elapsed':>8} | status",
        "-" * 102,
    ]
    for c in cells:
        loss = f"{c.final_loss:.4f}" if c.final_loss is not None else "—"
        et = f"{c.elapsed_seconds:.0f}s" if c.elapsed_seconds is not None else "—"
        lines.append(
            f"{c.cell_id:<18} | {c.peak_lr:>10.2e} | {c.init_std:>9.2e} | {c.tokens:>14,} | {loss:>11} | {et:>8} | {c.status}"
        )
    return "\n".join(lines)


def select_best(cells: list[SweepCell]) -> SweepCell | None:
    """Return the cell with the lowest final loss (None if no cells finished)."""
    done = [c for c in cells if c.status == "done" and c.final_loss is not None]
    if not done:
        return None
    return min(done, key=lambda c: c.final_loss)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-config", default=str(_REPO / "configs" / "wind_tunnel.yaml"))
    p.add_argument("--data-config", default=str(_REPO / "configs" / "data" / "pretrain_mix.yaml"))
    p.add_argument("--tokenizer-path", default="artifacts/tokenizer_v1.json")
    p.add_argument(
        "--tokens-per-cell",
        type=int,
        default=200_000_000,
        help="Tokens per sweep cell (default 200M; μP convention for 30M model).",
    )
    p.add_argument("--lr-grid", type=str, default=",".join(repr(x) for x in DEFAULT_LR_GRID),
                   help="Comma-separated list of peak_lr values.")
    p.add_argument("--init-grid", type=str, default=",".join(repr(x) for x in DEFAULT_INIT_GRID),
                   help="Comma-separated list of init_std values.")
    p.add_argument("--output", default="artifacts/wind_tunnel_plan.json")
    p.add_argument("--artifact-root", default="artifacts")
    p.add_argument("--execute", action="store_true",
                   help="Actually run each cell. Default is dry-run (print plan only).")
    p.add_argument("--collect", action="store_true",
                   help="Re-scan a previously written manifest's log files and update final_loss values.")
    args = p.parse_args()

    lr_grid = tuple(float(x) for x in args.lr_grid.split(","))
    init_grid = tuple(float(x) for x in args.init_grid.split(","))

    cells = build_grid(lr_grid, init_grid, tokens_per_cell=args.tokens_per_cell)
    manifest = SweepManifest(
        config_path=args.model_config,
        data_config_path=args.data_config,
        tokenizer_path=args.tokenizer_path,
        tokens_per_cell=args.tokens_per_cell,
        cells=cells,
    )

    if args.collect:
        # Re-parse logs from a previous --execute run.
        for c in manifest.cells:
            if c.run_log_path:
                c.final_loss = parse_final_loss(Path(c.run_log_path))
                c.status = "done" if c.final_loss is not None else "failed"
        print(report_grid(manifest.cells))
        best = select_best(manifest.cells)
        if best:
            print(f"\nbest: {best.cell_id}  peak_lr={best.peak_lr:.2e}  init_std={best.init_std:.2e}  loss={best.final_loss:.4f}")
        return 0

    if not args.execute:
        # Dry-run: write the plan + print each command.
        print(f"Wind-tunnel sweep plan — {len(cells)} cells")
        print(f"  model_config:  {args.model_config}")
        print(f"  data_config:   {args.data_config}")
        print(f"  tokenizer:     {args.tokenizer_path}")
        print(f"  tokens/cell:   {args.tokens_per_cell:,}")
        print(f"  output:        {args.output}")
        print()
        for c in cells:
            cmd = cell_command(
                c, args.model_config, args.data_config, args.tokenizer_path,
                checkpoint_root=str(Path(args.artifact_root) / "wind_tunnel" / c.cell_id / "ckpts"),
                log_path=str(Path(args.artifact_root) / "wind_tunnel" / c.cell_id / "train.log"),
            )
            print(f"# cell {c.cell_id}")
            print("  " + " \\\n  ".join(cmd))
            print()
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(manifest.to_dict(), indent=2))
        print(f"\nDry-run complete. Plan written to {args.output}.")
        print("Run with --execute to actually launch each cell.")
        return 0

    # Execute path: run each cell sequentially.
    artifact_root = Path(args.artifact_root)
    manifest.started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for c in manifest.cells:
        print(f"=== running cell {c.cell_id} (peak_lr={c.peak_lr:.2e}, init_std={c.init_std:.2e}) ===")
        run_cell(c, args.model_config, args.data_config, args.tokenizer_path, artifact_root)
        print(f"  status={c.status} loss={c.final_loss} elapsed={c.elapsed_seconds}s")
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(manifest.to_dict(), indent=2))
    manifest.completed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    Path(args.output).write_text(json.dumps(manifest.to_dict(), indent=2))

    print(report_grid(manifest.cells))
    best = select_best(manifest.cells)
    if best:
        print(f"\nbest: {best.cell_id}  peak_lr={best.peak_lr:.2e}  init_std={best.init_std:.2e}  loss={best.final_loss:.4f}")
        print(f"\nApply to pilot/base configs via:")
        print(f"  pilot_250m.yaml.lr_schedule.peak_lr = {best.peak_lr:.4e}")
        print(f"  pilot_250m.yaml.init_std            = {best.init_std:.4e}")
        print(f"  pilot_250m.yaml.mup.base_width      = 256  (matches wind_tunnel.yaml)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
