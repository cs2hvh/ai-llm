"""Regression tests for B6 (2026-05-12 audit): bad-batch quarantine writer.

The earlier audit recommended "identify and quarantine the offending
batch/doc" as the proper response to NaN, not just NaN-skip. This module
implements the writer; these tests verify it produces well-formed JSONL
suitable for post-mortem.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from myllm.training.quarantine import QuarantineWriter


def _make_batch(B: int = 4, S: int = 16):
    return {
        "input_ids": np.arange(B * S, dtype=np.int32).reshape(B, S),
        "labels": np.arange(B * S, dtype=np.int32).reshape(B, S) + 1,
        "segment_ids": np.zeros((B, S), dtype=np.int32),
        "loss_mask": np.ones((B, S), dtype=np.int32),
    }


# --------------------------------------------------------------------------- #
# Basic write + JSONL format
# --------------------------------------------------------------------------- #
class TestBasicWrite:
    def test_writes_jsonl_record(self, tmp_path):
        q = QuarantineWriter(path=tmp_path / "q.jsonl")
        q.write(step=4200, data_position=68_000_000, batch=_make_batch(),
                loss=float("nan"), reason="nan_skipped")
        assert q.incident_count == 1

        lines = (tmp_path / "q.jsonl").read_text().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["step"] == 4200
        assert record["data_position"] == 68_000_000
        assert record["reason"] == "nan_skipped"
        # NaN loss serializes as None (not the JSON-invalid NaN literal).
        assert record["loss"] is None or record["loss"] != record["loss"]

    def test_appends_multiple_incidents(self, tmp_path):
        q = QuarantineWriter(path=tmp_path / "q.jsonl")
        for i in range(3):
            q.write(step=4200 + i, data_position=68_000_000 + i,
                    batch=_make_batch(), loss=float("nan"))
        lines = (tmp_path / "q.jsonl").read_text().splitlines()
        assert len(lines) == 3
        for i, line in enumerate(lines):
            record = json.loads(line)
            assert record["step"] == 4200 + i
        assert q.incident_count == 3

    def test_creates_parent_directory(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c" / "q.jsonl"
        q = QuarantineWriter(path=nested)
        q.write(step=1, data_position=0, batch=_make_batch(), loss=1.0)
        assert nested.exists()


# --------------------------------------------------------------------------- #
# Token preview — head + tail of each row
# --------------------------------------------------------------------------- #
class TestTokenPreview:
    def test_input_ids_preview_captures_head_and_tail(self, tmp_path):
        q = QuarantineWriter(path=tmp_path / "q.jsonl", max_token_preview=4)
        batch = _make_batch(B=2, S=20)
        q.write(step=1, data_position=0, batch=batch, loss=float("nan"))

        record = json.loads((tmp_path / "q.jsonl").read_text())
        preview = record["input_ids_preview"]
        assert len(preview) == 2
        # Row 0: ids 0..19; head = [0,1,2,3], tail = [16,17,18,19]
        assert preview[0]["head"] == [0, 1, 2, 3]
        assert preview[0]["tail"] == [16, 17, 18, 19]
        # Row 1: ids 20..39
        assert preview[1]["head"] == [20, 21, 22, 23]
        assert preview[1]["tail"] == [36, 37, 38, 39]

    def test_short_sequence_tail_empty(self, tmp_path):
        """If the sequence is shorter than n, tail is empty (head covers all)."""
        q = QuarantineWriter(path=tmp_path / "q.jsonl", max_token_preview=32)
        batch = {"input_ids": np.arange(10, dtype=np.int32).reshape(1, 10)}
        q.write(step=1, data_position=0, batch=batch, loss=1.0)

        record = json.loads((tmp_path / "q.jsonl").read_text())
        assert record["input_ids_preview"][0]["head"] == list(range(10))
        assert record["input_ids_preview"][0]["tail"] == []

    def test_missing_input_ids_handled_gracefully(self, tmp_path):
        q = QuarantineWriter(path=tmp_path / "q.jsonl")
        q.write(step=1, data_position=0, batch={"labels": np.zeros((1, 4))}, loss=1.0)
        record = json.loads((tmp_path / "q.jsonl").read_text())
        assert record["input_ids_preview"] is None


# --------------------------------------------------------------------------- #
# Segment histogram — useful for finding doc-packing pathologies
# --------------------------------------------------------------------------- #
class TestSegmentHistogram:
    def test_segment_histogram_per_row(self, tmp_path):
        q = QuarantineWriter(path=tmp_path / "q.jsonl")
        batch = _make_batch(B=2, S=8)
        # Row 0: 3 docs (segments 0,0,0,1,1,2,2,2)
        batch["segment_ids"] = np.array([
            [0, 0, 0, 1, 1, 2, 2, 2],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ], dtype=np.int32)
        q.write(step=1, data_position=0, batch=batch, loss=float("nan"))

        record = json.loads((tmp_path / "q.jsonl").read_text())
        hist = record["segment_ids_histogram"]
        assert len(hist) == 2
        assert hist[0] == {"0": 3, "1": 2, "2": 3}
        assert hist[1] == {"0": 8}

    def test_missing_segment_ids_returns_none(self, tmp_path):
        q = QuarantineWriter(path=tmp_path / "q.jsonl")
        q.write(step=1, data_position=0,
                batch={"input_ids": np.zeros((1, 4), dtype=np.int32)},
                loss=1.0)
        record = json.loads((tmp_path / "q.jsonl").read_text())
        assert record["segment_ids_histogram"] is None


# --------------------------------------------------------------------------- #
# Robustness — never raise on bad input
# --------------------------------------------------------------------------- #
class TestRobustness:
    def test_garbage_batch_does_not_crash(self, tmp_path):
        """In production we'd rather log SOMETHING than crash the run."""
        q = QuarantineWriter(path=tmp_path / "q.jsonl")
        q.write(step=1, data_position=0, batch={"weird": "string"},
                loss=float("nan"))
        # Should have written a record, even if previews are None.
        record = json.loads((tmp_path / "q.jsonl").read_text())
        assert record["step"] == 1
        assert record["batch_shape"] is None  # no array shapes found
