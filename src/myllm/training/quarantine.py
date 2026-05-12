"""Bad-batch quarantine for forensic NaN debugging.

When ``train_step`` atomically reverts because loss or gradients went
non-finite (P0-1 fix), the loop should not just log + continue — it
should dump enough of the offending batch to disk that a post-mortem
can identify the poisonous document(s).

This module provides ``QuarantineWriter``: a thin JSONL-per-incident
recorder that captures:
    - step at which the NaN fired
    - batch shapes (B × S)
    - per-row first + last 32 token ids (so we can decode + inspect)
    - per-row segment_id histogram (so we can see if doc boundaries
      coincide with the spike)
    - data_position cursor (so we know where in the corpus the bad
      batch sat)
    - timestamp

The format is one JSON record per spike, append-mode. Small. Suitable
for tail-grepping during long runs.

B6 from 2026-05-12 audit. The dossier verdict was that we should
"identify and quarantine the offending batch/doc" rather than rely
on NaN-skip as a primary defense. This writer is that quarantine
tool.

Decoding the snippets back to text requires the tokenizer used during
training — a separate ``scripts/inspect_quarantine.py`` (Phase B follow-
up) will tokenizer.decode() each first/last-32 chunk for the operator.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from myllm.utils import get_logger

log = get_logger(__name__)


class QuarantineWriter:
    """Append-only JSONL writer for bad-batch incidents.

    Thread-safety: not thread-safe. The training loop runs single-threaded
    per pod, so this is fine.

    Usage:
        q = QuarantineWriter(path="artifacts/quarantine.jsonl")
        # ... inside the loop, after detecting nan_skipped > 0:
        q.write(step=int(state["step"]),
                data_position=int(state["data_position"]),
                batch=batch,
                loss=float(metrics["loss"]),
                reason="nan_skipped")
    """

    def __init__(self, path: str | Path, max_token_preview: int = 32):
        self.path = Path(path)
        self.max_token_preview = int(max_token_preview)
        self._count = 0
        # Quarantine is a forensic aid, not load-bearing. If the parent
        # directory can't be created (read-only volume, permission denied,
        # out of inodes), log + degrade — never let init kill the run.
        # The write() method already wraps the actual append in try/except.
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._disabled = False
        except OSError as e:
            log.error(
                "quarantine_disabled_parent_unwritable",
                path=str(self.path),
                error=str(e),
            )
            self._disabled = True

    @property
    def incident_count(self) -> int:
        return self._count

    def write(
        self,
        *,
        step: int,
        data_position: int,
        batch: dict[str, Any],
        loss: float,
        reason: str = "nan_skipped",
    ) -> None:
        """Record one incident. Mostly best-effort — never raises on bad
        input (we're already in an unhappy path)."""
        if self._disabled:
            return
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "step": int(step),
            "data_position": int(data_position),
            "reason": str(reason),
            "loss": _safe_float(loss),
            "batch_shape": _batch_shape(batch),
            "input_ids_preview": _token_preview(
                batch.get("input_ids"), n=self.max_token_preview
            ),
            "segment_ids_histogram": _segment_histogram(batch.get("segment_ids")),
        }
        try:
            with open(self.path, "a") as f:
                f.write(json.dumps(record) + "\n")
            self._count += 1
        except Exception as e:  # noqa: BLE001
            log.error(
                "quarantine_write_failed",
                step=step,
                error=str(e),
            )


def _safe_float(x: Any) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _batch_shape(batch: dict[str, Any]) -> dict[str, list[int]] | None:
    """Return {field: shape} for keys with array-like values."""
    if not isinstance(batch, dict):
        return None
    out: dict[str, list[int]] = {}
    for k, v in batch.items():
        shape = getattr(v, "shape", None)
        if shape is not None:
            try:
                out[k] = list(int(d) for d in shape)
            except (TypeError, ValueError):
                continue
    return out or None


def _token_preview(arr: Any, *, n: int) -> list[dict[str, list[int]]] | None:
    """For each row in input_ids, return its first n + last n token ids.
    These are short enough to decode + inspect manually in a post-mortem."""
    if arr is None:
        return None
    try:
        # Coerce to a python list of lists — works for numpy + jax arrays.
        # We import numpy lazily; if it's not present, return None.
        import numpy as np

        a = np.asarray(arr)
        if a.ndim != 2:
            return None
        rows = []
        for row in a:
            row_list = [int(t) for t in row]
            head = row_list[:n]
            tail = row_list[-n:] if len(row_list) > n else []
            rows.append({"head": head, "tail": tail})
        return rows
    except Exception:  # noqa: BLE001
        return None


def _segment_histogram(seg: Any) -> list[dict[str, int]] | None:
    """For each row in segment_ids, return {unique_segment_id: count}.
    Useful to spot rows that pack many small docs (potential pathology)."""
    if seg is None:
        return None
    try:
        import numpy as np

        a = np.asarray(seg)
        if a.ndim != 2:
            return None
        rows = []
        for row in a:
            uniq, counts = np.unique(row, return_counts=True)
            rows.append({int(u): int(c) for u, c in zip(uniq, counts)})
        return rows
    except Exception:  # noqa: BLE001
        return None
