#!/usr/bin/env python3
"""Parallel-fan-out runner for B2 per-source corpus builds.

Single-source `build_packed_corpus.py` is HF-stream-bound (~10-17K tok/sec
end-to-end on this server, regardless of filter/dedupe settings — see
2026-05-12 probe). Each source's HF stream is single-threaded, so the
right scaling is independent processes — one per source.

This runner:
  - Reads the source list from configs/data/pretrain_mix.yaml
  - Spawns one subprocess per source (capped at --max-parallel)
  - Each subprocess runs scripts/build_packed_corpus.py with R2 streaming
    + delete-local (so disk stays bounded)
  - Logs per-source stdout to artifacts/build_logs/<source>.log
  - Tracks completion + aggregates summary JSON
  - Returns non-zero if any source fails

Wall-time estimate (on 13 sources, ~13K tok/sec each):
  ~13× speedup vs sequential. 100B tokens drops from ~89 days → ~7 days.
  10B drops from ~9 days → ~17 hours. 1B drops from ~21 hours → ~1.7 hr.

Use:
    python scripts/run_parallel_builds.py \\
        --output-root /workspace/corpus/sources \\
        --r2-prefix corpus_v1/sources \\
        --sample-limit 50000 \\
        --max-parallel 12 \\
        --delete-local-after-upload
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parents[1]


def _resolve_source_list(pretrain_mix_path: Path) -> list[dict]:
    cfg = yaml.safe_load(pretrain_mix_path.read_text())
    return cfg.get("sources", [])


def _unique_source_id(entry: dict) -> str:
    """Stable unique id per (dataset, config_name, split) tuple.

    Catches the multi-config-source collision class (mc4 × 5 configs all
    naming themselves "mc4" → 5 processes fighting over the same output
    dir + log file). Each row in pretrain_mix.yaml's sources list maps
    to one unique build, regardless of duplicate dataset names.
    """
    dataset = entry["dataset"]
    short = dataset.split("/")[-1].lower().replace(".", "_").replace("-", "_")
    suffix_parts: list[str] = []
    cfg = entry.get("config_name")
    if cfg:
        suffix_parts.append(str(cfg).replace(".", "_").replace("/", "_"))
    split = entry.get("split")
    if split and split != "train":
        suffix_parts.append(f"split_{split}")
    if suffix_parts:
        return f"{short}_{'_'.join(suffix_parts)}"
    return short


def _write_per_source_yaml(
    entry: dict, base_cfg: dict, tmp_dir: Path, unique_id: str,
) -> Path:
    """Write a single-source pretrain_mix.yaml for one subprocess to consume.

    Avoids ambiguity when multiple yaml rows have the same `dataset` field
    (mc4 with different config_name). build_packed_corpus.py looks up by
    dataset name; with only one entry in the temp yaml, the lookup is
    unambiguous.
    """
    cfg = {k: v for k, v in base_cfg.items() if k != "sources"}
    cfg["sources"] = [entry]
    out = tmp_dir / f"pretrain_mix_{unique_id}.yaml"
    out.write_text(yaml.safe_dump(cfg))
    return out


def _per_source_log_path(log_root: Path, unique_id: str) -> Path:
    return log_root / f"{unique_id}.log"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pretrain-mix-config",
                   default=str(_REPO / "configs" / "data" / "pretrain_mix.yaml"))
    p.add_argument("--model-config",
                   default=str(_REPO / "configs" / "base_1b.yaml"),
                   help="For sequence_length resolution.")
    p.add_argument("--tokenizer-path",
                   default=str(_REPO / "artifacts" / "tokenizer_v1.json"))
    p.add_argument("--output-root", required=True,
                   help="Per-source outputs go to <output-root>/<source-id>/")
    p.add_argument("--r2-prefix", default=None,
                   help="If set, each source's shards stream to "
                        "s3://$S3_BUCKET/<r2-prefix>/<source-id>/...")
    p.add_argument("--delete-local-after-upload", action="store_true")
    p.add_argument("--sample-limit", type=int, default=None,
                   help="Per-source doc cap. Forwarded to each subprocess.")
    p.add_argument("--target-tokens-per-source", type=int, default=None,
                   help="Per-source TOKEN budget proportional to share. "
                        "Each source's --target-tokens is set to "
                        "<this> * <source.share>. Bounds wall time when "
                        "doc sizes vary (pg19 books vs FineWeb-Edu pages).")
    p.add_argument("--max-parallel", type=int, default=12,
                   help="Max concurrent source processes. Default 12 — "
                        "we have 13 sources + 128 CPU cores.")
    p.add_argument("--sequence-length", type=int, default=None)
    p.add_argument("--sequences-per-shard", type=int, default=65536)
    p.add_argument("--revision-id", default=None,
                   help="Common revision-id label (default: build-YYYYMMDD).")
    p.add_argument("--no-decontam", action="store_true",
                   help="Skip per-source decontamination. Forwarded.")
    p.add_argument("--no-dedupe", action="store_true",
                   help="Skip per-source MinHash dedupe. Forwarded.")
    p.add_argument("--no-filters", action="store_true",
                   help="Skip filter chain. Forwarded.")
    p.add_argument("--skip-sources", default="",
                   help="Comma-separated dataset names to skip "
                        "(e.g. bigcode/the-stack-v2 if its loader is broken).")
    p.add_argument("--log-dir", default=None,
                   help="Per-source stdout logs (default: artifacts/build_logs).")
    args = p.parse_args()

    pretrain_mix = Path(args.pretrain_mix_config)
    base_cfg = yaml.safe_load(pretrain_mix.read_text())
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    log_root = Path(args.log_dir or (_REPO / "artifacts" / "build_logs"))
    log_root.mkdir(parents=True, exist_ok=True)
    # Temp dir for per-source yamls (deleted on completion).
    import tempfile
    tmp_root = Path(tempfile.mkdtemp(prefix="parallel_build_"))
    revision = args.revision_id or f"build-{time.strftime('%Y%m%d')}"
    skip = {s.strip() for s in args.skip_sources.split(",") if s.strip()}

    sources = _resolve_source_list(pretrain_mix)
    sources = [s for s in sources if s["dataset"] not in skip]
    if not sources:
        print("ERROR: no sources to build after skip filter", file=sys.stderr)
        return 2

    # Compute unique source_id per row. Detect (and refuse) duplicates as
    # a defensive measure — should never happen with _unique_source_id's
    # disambiguation, but useful insurance.
    rows: list[dict] = []
    seen_ids: dict[str, dict] = {}
    for entry in sources:
        uid = _unique_source_id(entry)
        if uid in seen_ids:
            print(
                f"ERROR: duplicate unique_source_id {uid!r} for entries:\n"
                f"  {seen_ids[uid]}\n  {entry}\n"
                "Add a config_name or split to disambiguate.",
                file=sys.stderr,
            )
            return 2
        seen_ids[uid] = entry
        rows.append({"entry": entry, "unique_id": uid})

    print(f"=== parallel build: {len(rows)} rows, max_parallel={args.max_parallel}")
    for r in rows:
        e = r["entry"]
        suffix = ""
        if e.get("config_name"):
            suffix = f" ({e['config_name']})"
        print(f"  - {r['unique_id']:<40} ← {e['dataset']}{suffix} "
              f"(share={e.get('share', 0):.3f})")

    # Build the per-source argv template — uses a single-source temp yaml
    # so the build_packed_corpus.py CLI's source-lookup is unambiguous
    # (avoids the mc4 multi-config collision class).
    def _make_argv(row: dict) -> list[str]:
        entry = row["entry"]
        uid = row["unique_id"]
        per_source_yaml = _write_per_source_yaml(entry, base_cfg, tmp_root, uid)
        argv = [
            sys.executable,
            str(_REPO / "scripts" / "build_packed_corpus.py"),
            "--source", entry["dataset"],
            "--source-id", uid,
            "--pretrain-mix-config", str(per_source_yaml),
            "--model-config", str(args.model_config),
            "--tokenizer-path", str(args.tokenizer_path),
            "--output-root", str(output_root),
            "--sequences-per-shard", str(args.sequences_per_shard),
            "--revision-id", revision,
        ]
        if args.sequence_length is not None:
            argv.extend(["--sequence-length", str(args.sequence_length)])
        if args.sample_limit is not None:
            argv.extend(["--sample-limit", str(args.sample_limit)])
        if args.target_tokens_per_source is not None:
            # Allocate per-source tokens proportional to share.
            per_source = int(args.target_tokens_per_source * float(entry.get("share", 0)))
            if per_source > 0:
                argv.extend(["--target-tokens", str(per_source)])
        if args.r2_prefix:
            argv.extend(["--r2-prefix", args.r2_prefix])
        if args.delete_local_after_upload:
            argv.append("--delete-local-after-upload")
        if args.no_decontam:
            argv.append("--no-decontam")
        if args.no_dedupe:
            argv.append("--no-dedupe")
        if args.no_filters:
            argv.append("--no-filters")
        return argv

    # Schedule. Each row becomes a Popen; we cap concurrency at max_parallel.
    pending = list(rows)
    running: list[tuple[dict, subprocess.Popen, Path]] = []
    results: list[dict] = []
    t_start = time.time()

    while pending or running:
        # Launch up to max_parallel.
        while pending and len(running) < args.max_parallel:
            row = pending.pop(0)
            uid = row["unique_id"]
            log_path = _per_source_log_path(log_root, uid)
            print(f"[+] launching: {uid}  →  log: {log_path}")
            proc = subprocess.Popen(
                _make_argv(row),
                stdout=open(log_path, "wb"),
                stderr=subprocess.STDOUT,
                env=os.environ,
            )
            running.append((row, proc, log_path))

        # Poll running.
        time.sleep(2)
        still_running: list[tuple[dict, subprocess.Popen, Path]] = []
        for row, proc, log_path in running:
            rc = proc.poll()
            if rc is None:
                still_running.append((row, proc, log_path))
                continue
            entry = row["entry"]
            uid = row["unique_id"]
            wall = time.time() - t_start
            print(f"[{'✓' if rc == 0 else '✗'}] done: {uid}  rc={rc}  "
                  f"wall={wall:.0f}s")
            # Parse the per-source JSON summary printed at the end of stdout.
            try:
                log_text = log_path.read_text()
                last_open = log_text.rfind("{")
                summary = json.loads(log_text[last_open:].strip())
            except Exception:  # noqa: BLE001
                summary = {"error": "could not parse summary", "log": str(log_path)}
            results.append({
                "unique_id": uid,
                "dataset": entry["dataset"],
                "config_name": entry.get("config_name"),
                "share": float(entry.get("share", 0)),
                "returncode": rc,
                "log": str(log_path),
                "summary": summary,
            })
        running = still_running

    wall_total = time.time() - t_start
    n_pass = sum(1 for r in results if r["returncode"] == 0)
    n_fail = len(results) - n_pass

    report = {
        "wall_seconds": round(wall_total, 1),
        "n_sources_total": len(results),
        "n_pass": n_pass,
        "n_fail": n_fail,
        "results": results,
    }
    print()
    print("=== summary ===")
    print(f"wall: {wall_total:.0f}s  ({wall_total / 60:.1f} min)")
    print(f"pass: {n_pass}/{len(results)}")
    if n_fail:
        print(f"FAIL: {n_fail} sources had non-zero exit codes — check logs.")
    # Print + persist the report.
    report_path = log_root / f"parallel_build_summary_{int(t_start)}.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"full report: {report_path}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
