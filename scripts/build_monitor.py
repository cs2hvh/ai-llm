#!/usr/bin/env python3
"""Live monitor for a parallel-fan-out corpus build.

Polls per-source log files written by ``scripts/run_parallel_builds.py``
+ disk + R2 + process state every N seconds, prints a refreshable
status table. Designed to run alongside the build in a separate
terminal (or tmux/screen pane):

    # Terminal 1:
    nohup python scripts/run_parallel_builds.py \\
        --output-root /workspace/corpus_v1/sources \\
        --sample-limit 50000 \\
        --max-parallel 12 \\
        --r2-prefix corpus_v1/sources \\
        --delete-local-after-upload \\
        > artifacts/build_logs/runner.log 2>&1 &

    # Terminal 2:
    python scripts/build_monitor.py --log-dir artifacts/build_logs

The per-source log files (artifacts/build_logs/<safe-name>.log) are
structlog JSON streams. The monitor parses key events:
  - hf_stream_open                 → source has started
  - build_one_source_start         → tokenization beginning
  - packed_corpus_shard_close      → shard N closed (count + token rate)
  - packed_corpus_shard_uploaded   → R2 upload completed
  - build_one_source_done          → source finished (stats in record)
  - any "level": "error"           → red-flag

Status table refreshes every ``--interval`` seconds (default 5).
Press Ctrl-C to exit (the build keeps running in its terminal).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SourceState:
    """Per-source rollup parsed from the source's log stream."""

    name: str
    started_at: float | None = None
    finished_at: float | None = None
    last_seen_ts: float | None = None
    n_shards_closed: int = 0
    n_shards_uploaded: int = 0
    tokens_so_far: int = 0
    docs_seen: int = 0
    docs_kept: int = 0
    docs_filtered: int = 0
    docs_deduped: int = 0
    docs_contaminated: int = 0
    errors: list[str] = field(default_factory=list)
    final_summary: dict | None = None
    process_alive: bool | None = None

    @property
    def status(self) -> str:
        if self.errors and self.final_summary is None:
            return "ERR"
        if self.final_summary is not None:
            return "DONE"
        if self.started_at is None:
            return "QUEUE"
        return "RUN"

    @property
    def runtime_seconds(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.finished_at or time.time()
        return end - self.started_at

    @property
    def tok_per_sec(self) -> float:
        rt = self.runtime_seconds
        if rt <= 0:
            return 0.0
        return self.tokens_so_far / rt


def _parse_log_file(path: Path, state: SourceState) -> None:
    """Read the (possibly partial) log file + update state in place.

    The log is a structlog JSON stream — each line is one event dict.
    Also tolerates non-JSON lines (Python traceback frames, etc.).
    """
    if not path.exists():
        return
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if not (line.startswith("{") and line.endswith("}")):
                    if "Traceback" in line or "Error" in line or "Exception" in line:
                        state.errors.append(line[:240])
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                state.last_seen_ts = time.time()
                lvl = e.get("level", "info")
                evt = e.get("event", "")
                if lvl == "error":
                    state.errors.append(evt + ": " + json.dumps(
                        {k: v for k, v in e.items()
                         if k not in ("level", "timestamp", "event")}
                    )[:240])
                if evt == "hf_stream_open" and state.started_at is None:
                    state.started_at = time.time()
                elif evt == "build_one_source_start":
                    state.started_at = state.started_at or time.time()
                elif evt == "packed_corpus_shard_close":
                    state.n_shards_closed += 1
                    state.tokens_so_far += int(e.get("total_tokens", 0))
                elif evt == "packed_corpus_shard_uploaded":
                    state.n_shards_uploaded += 1
                elif evt == "build_one_source_done":
                    state.finished_at = time.time()
                    state.docs_seen = int(e.get("docs_seen", state.docs_seen))
                    state.docs_kept = int(e.get("docs_kept", state.docs_kept))
                    state.docs_filtered = int(e.get("docs_filtered", state.docs_filtered))
                    state.docs_deduped = int(e.get("docs_deduped", state.docs_deduped))
                    state.docs_contaminated = int(e.get("docs_contaminated",
                                                        state.docs_contaminated))
        # Try to capture the final JSON summary that build_packed_corpus.py
        # prints to stdout at the end (it's pretty-printed JSON, not structlog).
        if state.finished_at is not None and state.final_summary is None:
            text = path.read_text()
            last_open = text.rfind("\n{\n")  # multi-line JSON block
            if last_open != -1:
                try:
                    state.final_summary = json.loads(text[last_open:].strip())
                except json.JSONDecodeError:
                    pass
    except OSError:
        pass


def _disk_usage(path: Path) -> tuple[float, float, float]:
    """Return (used_gb, free_gb, used_fraction) for the filesystem at ``path``."""
    try:
        usage = shutil.disk_usage(path)
        used = (usage.total - usage.free) / 1e9
        free = usage.free / 1e9
        frac = (usage.total - usage.free) / usage.total
        return used, free, frac
    except OSError:
        return 0.0, 0.0, 0.0


def _format_int(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _format_secs(s: float) -> str:
    if s < 60:
        return f"{int(s)}s"
    if s < 3600:
        return f"{int(s // 60)}m{int(s % 60):02d}s"
    return f"{int(s // 3600)}h{int((s % 3600) // 60):02d}m"


def _format_rate(tps: float) -> str:
    if tps >= 1_000_000:
        return f"{tps / 1_000_000:.2f}M/s"
    if tps >= 1_000:
        return f"{tps / 1_000:.1f}K/s"
    return f"{tps:.0f}/s"


def _print_status(
    states: dict[str, SourceState],
    *,
    output_root: Path | None,
    elapsed_total: float,
) -> None:
    # Clear screen with ANSI escape (best-effort).
    print("\033[2J\033[H", end="", flush=True)

    print(f"=== MyLLM corpus build monitor ===  elapsed: {_format_secs(elapsed_total)}")
    print()

    if not states:
        print("(no source logs found yet — runner may still be starting)")
        return

    # Header.
    print(f"{'source':<48} {'status':<6} {'time':<8} {'shards':<8} "
          f"{'tokens':>10} {'tok/sec':>10}")
    print("-" * 100)

    total_tokens = 0
    total_shards = 0
    n_done = 0
    n_err = 0
    n_run = 0

    for name in sorted(states):
        s = states[name]
        runtime = s.runtime_seconds
        line = (
            f"{name[:47]:<48} {s.status:<6} {_format_secs(runtime):>8} "
            f"{s.n_shards_uploaded:>3}/{s.n_shards_closed:<4} "
            f"{_format_int(s.tokens_so_far):>10} "
            f"{_format_rate(s.tok_per_sec):>10}"
        )
        # Status badges with ANSI color (best-effort; terminals without color
        # ignore the escape codes).
        if s.status == "DONE":
            line = f"\033[32m{line}\033[0m"  # green
            n_done += 1
        elif s.status == "ERR":
            line = f"\033[31m{line}\033[0m"  # red
            n_err += 1
        elif s.status == "RUN":
            n_run += 1
        print(line)
        total_tokens += s.tokens_so_far
        total_shards += s.n_shards_closed

    # Aggregate footer.
    print("-" * 100)
    aggregate_rate = total_tokens / elapsed_total if elapsed_total > 0 else 0.0
    print(f"{'TOTAL':<48} {f'{n_run}/{n_done}/{n_err}':<6} "
          f"{_format_secs(elapsed_total):>8} "
          f"{total_shards:>8} "
          f"{_format_int(total_tokens):>10} "
          f"{_format_rate(aggregate_rate):>10}")
    print("              (status = running/done/error)")

    # Disk footer.
    if output_root and output_root.exists():
        used, free, frac = _disk_usage(output_root)
        bar_w = 40
        filled = int(bar_w * frac)
        bar = "█" * filled + "░" * (bar_w - filled)
        color = "\033[33m" if frac > 0.85 else ""
        reset = "\033[0m" if color else ""
        print()
        print(f"disk @ {output_root}: {color}[{bar}]{reset}  "
              f"used={used:.0f} GB  free={free:.0f} GB  ({frac * 100:.0f}%)")

    # Errors.
    errs = [(name, s.errors[-1]) for name, s in states.items() if s.errors]
    if errs:
        print()
        print("\033[31m=== recent errors ===\033[0m")
        for name, err in errs[-5:]:
            print(f"  {name}: {err}")

    print()
    print(f"refresh: every {INTERVAL}s.  Ctrl-C to exit (build keeps running).")


# Global for the print fn — set in main().
INTERVAL = 5


def main() -> int:
    global INTERVAL
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--log-dir", default="artifacts/build_logs",
                   help="Directory where run_parallel_builds.py writes "
                        "per-source logs.")
    p.add_argument("--output-root", default=None,
                   help="Corpus output root (for disk usage display).")
    p.add_argument("--interval", type=int, default=5,
                   help="Refresh interval in seconds.")
    p.add_argument("--once", action="store_true",
                   help="Print a single snapshot and exit (no refresh loop).")
    args = p.parse_args()

    INTERVAL = args.interval

    log_dir = Path(args.log_dir).resolve()
    output_root = Path(args.output_root).resolve() if args.output_root else None
    if not log_dir.exists():
        print(f"log dir not found: {log_dir}", file=sys.stderr)
        print("Has the parallel build started? Re-run when logs exist.")
        return 2

    start = time.time()
    states: dict[str, SourceState] = {}

    try:
        while True:
            # Discover per-source log files. The parallel runner uses
            # <safe-name>.log where safe-name = dataset.replace('/', '__').
            for path in sorted(log_dir.glob("*.log")):
                # Skip the runner's own master log.
                if path.name == "runner.log":
                    continue
                # Restore the original dataset name (best-effort).
                source_name = path.stem.replace("__", "/")
                state = states.setdefault(source_name, SourceState(name=source_name))
                _parse_log_file(path, state)

            elapsed = time.time() - start
            _print_status(states, output_root=output_root, elapsed_total=elapsed)

            if args.once:
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nexiting monitor — build continues in its own terminal.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
