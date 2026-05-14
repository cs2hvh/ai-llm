#!/usr/bin/env python3
"""Inspect quarantine.jsonl from a pilot training run.

Quarantine writer (src/myllm/training/quarantine.py) dumps one JSON record
per NaN-skipped batch. This inspector parses those records and surfaces:

    - total event count, by reason
    - cluster detection (consecutive steps <100 apart → likely a poisonous
      contiguous stretch of corpus)
    - per-event data_position → source attribution (which source's doc
      caused each NaN), via the composed corpus's seq_meta.arrow
    - decoded text previews of the head/tail tokens (so an operator can
      eyeball the actual content)

Usage:
    # Minimal — just events:
    python scripts/inspect_quarantine.py /workspace/ckpt/pilot-250m-v1/quarantine.jsonl

    # With source attribution:
    python scripts/inspect_quarantine.py \\
        /workspace/ckpt/pilot-250m-v1/quarantine.jsonl \\
        --corpus-root /workspace/corpus_pilot_train

    # With decoded text:
    python scripts/inspect_quarantine.py \\
        /workspace/ckpt/pilot-250m-v1/quarantine.jsonl \\
        --corpus-root /workspace/corpus_pilot_train \\
        --tokenizer artifacts/tokenizer_v1.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("quarantine_file", help="Path to quarantine.jsonl")
    ap.add_argument("--corpus-root", help="Composed corpus dir (for data_position → source mapping)")
    ap.add_argument("--tokenizer", help="Tokenizer JSON path (for decoding token previews)")
    ap.add_argument("--cluster-threshold", type=int, default=100,
                    help="Events within this many steps count as a cluster (default 100)")
    ap.add_argument("--show-last", type=int, default=10,
                    help="How many most-recent events to show in detail (default 10)")
    args = ap.parse_args()

    qfile = Path(args.quarantine_file)
    if not qfile.exists():
        print(f"Quarantine file not found: {qfile}")
        print("(File is created on first NaN event; absent = no NaNs yet — that's good.)")
        return 0

    events = []
    with open(qfile) as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[warn] line {ln} malformed: {e}", file=sys.stderr)

    if not events:
        print("Quarantine file is empty — no NaN events recorded.")
        return 0

    print(f"=== {len(events)} quarantine events from {qfile} ===\n")
    _print_summary(events)
    _print_clusters(events, args.cluster_threshold)
    _print_recent(events, args.show_last)

    if args.corpus_root:
        _print_source_attribution(events, Path(args.corpus_root))

    if args.tokenizer:
        _print_token_decodes(events, Path(args.tokenizer), args.show_last)

    return 0


def _print_summary(events: list[dict]) -> None:
    by_reason = Counter(e.get("reason", "unknown") for e in events)
    print("By reason:")
    for r, n in by_reason.most_common():
        print(f"  {r:30s}: {n}")

    steps = sorted(int(e["step"]) for e in events if "step" in e)
    if steps:
        first = steps[0]
        last = steps[-1]
        span = max(1, last - first)
        rate = len(steps) / span * 1000
        print(f"\nStep range: {first} → {last} (span {span} steps)")
        print(f"Rate: {rate:.1f} events per 1000 steps")
        if rate > 5:
            print("  ⚠️  Rate exceeds 5/1000 threshold — investigate sources below")


def _print_clusters(events: list[dict], threshold: int) -> None:
    steps = sorted(int(e["step"]) for e in events if "step" in e)
    if len(steps) < 2:
        return
    clusters: list[list[int]] = []
    cur: list[int] = [steps[0]]
    for s in steps[1:]:
        if s - cur[-1] <= threshold:
            cur.append(s)
        else:
            if len(cur) >= 2:
                clusters.append(cur)
            cur = [s]
    if len(cur) >= 2:
        clusters.append(cur)

    if not clusters:
        print(f"\nNo clusters (no ≥2 events within {threshold} steps of each other).")
        return
    print(f"\nClusters (≥2 events within {threshold} steps):")
    for c in clusters:
        print(f"  steps {c[0]}..{c[-1]}: {len(c)} events")


def _print_recent(events: list[dict], n: int) -> None:
    print(f"\nLast {min(n, len(events))} events:")
    for e in events[-n:]:
        step = e.get("step", "?")
        dp = e.get("data_position", "?")
        loss = e.get("loss")
        loss_str = f"{loss}" if loss is None else f"{float(loss):.2f}"
        bshape = e.get("batch_shape", {})
        shape_str = ",".join(f"{k}={v}" for k, v in (bshape or {}).items() if k in ("input_ids",))
        print(f"  step {step:>6} | data_position {dp:>15} | loss {loss_str:>8} | {shape_str}")


def _print_source_attribution(events: list[dict], corpus_root: Path) -> None:
    """Map each event's data_position to the source that produced it."""
    top_manifest_path = corpus_root / "manifest.json"
    if not top_manifest_path.exists():
        print(f"\n[source mapping skipped: {top_manifest_path} not found]")
        return
    top_manifest = json.loads(top_manifest_path.read_text())
    seq_len = int(top_manifest["sequence_length"])
    seqs_per_shard = int(top_manifest["sequences_per_shard"])
    n_shards = int(top_manifest["n_shards"])

    try:
        import pyarrow.ipc as ipc  # noqa: F401
    except ImportError:
        print("\n[source mapping skipped: pyarrow not installed]")
        return

    # Cache: shard_id → list[source_id] keyed by sequence-in-shard
    shard_cache: dict[int, list[str]] = {}

    def get_source(seq_id: int) -> tuple[int, int, str | None]:
        shard_id = seq_id // seqs_per_shard
        in_shard = seq_id % seqs_per_shard
        if shard_id >= n_shards:
            return shard_id, in_shard, None
        if shard_id not in shard_cache:
            try:
                from pyarrow import ipc as _ipc
                p = corpus_root / f"shard-{shard_id:06d}" / "seq_meta.arrow"
                if not p.exists():
                    shard_cache[shard_id] = []
                else:
                    with _ipc.open_file(p) as reader:
                        tbl = reader.read_all()
                        # Composed corpus schema: each packed sequence is a MIX
                        # of one or more sources (a packer fills 8193 tokens
                        # from whichever doc fits next, then crosses doc
                        # boundaries marked by EOS). source_mix_keys[i] +
                        # source_mix_values[i] give per-source token counts
                        # for sequence i. We take the source with the most
                        # tokens as the dominant source.
                        keys_col = tbl.column("source_mix_keys") if "source_mix_keys" in tbl.column_names else None
                        vals_col = tbl.column("source_mix_values") if "source_mix_values" in tbl.column_names else None
                        if keys_col is None or vals_col is None:
                            shard_cache[shard_id] = []
                        else:
                            keys_list = keys_col.to_pylist()
                            vals_list = vals_col.to_pylist()
                            attribution = []
                            for keys, vals in zip(keys_list, vals_list):
                                if not keys:
                                    attribution.append("<empty>")
                                else:
                                    # dominant = argmax(vals)
                                    best_i = max(range(len(keys)), key=lambda i: (vals[i] if i < len(vals) else 0))
                                    dom = keys[best_i]
                                    if len(keys) == 1:
                                        attribution.append(dom)
                                    else:
                                        # multi-source pack — annotate with the mix
                                        mix_str = ",".join(f"{k}={v}" for k, v in zip(keys, vals))
                                        attribution.append(f"{dom} [mix: {mix_str}]")
                            shard_cache[shard_id] = attribution
            except Exception as e:  # noqa: BLE001
                print(f"[shard-{shard_id:06d} seq_meta read failed: {e}]", file=sys.stderr)
                shard_cache[shard_id] = []
        srcs = shard_cache[shard_id]
        return shard_id, in_shard, (srcs[in_shard] if in_shard < len(srcs) else None)

    print(f"\n=== Source attribution (data_position → seq_id → source) ===")
    print(f"sequence_length={seq_len}, sequences_per_shard={seqs_per_shard}, n_shards={n_shards}\n")

    counts: Counter = Counter()
    dom_counts: Counter = Counter()  # dominant-source only, strips [mix: ...] annotation
    multi_src_count = 0
    for e in events:
        dp = e.get("data_position")
        step = e.get("step", "?")
        if dp is None:
            continue
        seq_id = int(dp) // seq_len
        shard_id, in_shard, src = get_source(seq_id)
        counts[src or "<unmapped>"] += 1
        # Extract just the dominant source name (strip "[mix: ...]" suffix)
        dom = (src or "<unmapped>").split(" [mix:")[0]
        dom_counts[dom] += 1
        if src and "[mix:" in src:
            multi_src_count += 1
        print(f"  step {step:>6} | seq_id {seq_id:>7} | shard-{shard_id:06d}[{in_shard:>5}] | {src}")

    print(f"\nDominant-source attribution summary:")
    for src, n in dom_counts.most_common():
        print(f"  {str(src):40s}: {n}")
    if multi_src_count:
        print(f"\n({multi_src_count}/{len(events)} events were multi-source packs — the dominant source above is the one with the most tokens in the packed sequence)")


def _print_token_decodes(events: list[dict], tokenizer_path: Path, n: int) -> None:
    try:
        from tokenizers import Tokenizer
    except ImportError:
        print("\n[token decoding skipped: 'tokenizers' lib not installed]")
        return
    if not tokenizer_path.exists():
        print(f"\n[token decoding skipped: tokenizer not found at {tokenizer_path}]")
        return
    tk = Tokenizer.from_file(str(tokenizer_path))
    print(f"\n=== Decoded token previews (last {min(n, len(events))} events) ===")
    for e in events[-n:]:
        step = e.get("step", "?")
        preview = e.get("input_ids_preview") or []
        if not preview:
            continue
        print(f"\n  step {step}:")
        for i, row in enumerate(preview[:4]):  # show first 4 batch rows max
            head = row.get("head", [])
            tail = row.get("tail", [])
            head_text = tk.decode(head, skip_special_tokens=False) if head else ""
            tail_text = tk.decode(tail, skip_special_tokens=False) if tail else ""
            print(f"    row {i} head: {head_text[:200]!r}")
            print(f"    row {i} tail: {tail_text[:200]!r}")


if __name__ == "__main__":
    sys.exit(main())
