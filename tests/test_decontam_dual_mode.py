"""Tests for DualModeDecontaminationIndex (2026-05-13, reviewer Q7).

Verifies that:
  1. The wrapper presents the same scan_document interface as a single
     index, so existing pipeline code works without changes.
  2. Matches are the UNION of both modes (primary + secondary).
  3. Either mode can be None.
  4. The secondary mode's per-benchmark counts get tracked separately
     in the report without double-counting the scan-count.
"""
from __future__ import annotations

import pytest

from myllm.data.decontamination import (
    DecontaminationConfig,
    DecontaminationIndex,
    DecontaminationReport,
    DualModeDecontaminationIndex,
)


def _make_idx(ngram_size: int, prompts: dict[str, list[str]]) -> DecontaminationIndex:
    idx = DecontaminationIndex(DecontaminationConfig(ngram_size=ngram_size))
    for bench_id, ps in prompts.items():
        idx.add_benchmark(bench_id, ps)
    return idx


class TestDualModeBasics:
    def test_requires_at_least_one_index(self):
        with pytest.raises(ValueError, match="at least one"):
            DualModeDecontaminationIndex(primary=None, secondary=None)

    def test_primary_only_works(self):
        # Note: benchmark prompts must be at least `ngram_size` words long
        # for any n-grams to be extracted.
        idx_8 = _make_idx(8, {"bench_a": [
            "the quick brown fox jumps over the lazy dog today"
        ]})
        dual = DualModeDecontaminationIndex(primary=idx_8)
        m = dual.scan_document("the quick brown fox jumps over the lazy dog today indeed")
        assert m.get("bench_a", 0) > 0

    def test_secondary_only_works(self):
        # Pure secondary (e.g., 13-gram alone) is supported. Prompt needs
        # >= 13 words.
        idx_13 = _make_idx(13, {
            "bench_b": [
                "this is a moderately long sentence that fingerprints a "
                "benchmark uniquely across the corpus"
            ]
        })
        dual = DualModeDecontaminationIndex(secondary=idx_13)
        m = dual.scan_document(
            "this is a moderately long sentence that fingerprints a "
            "benchmark uniquely across the corpus today"
        )
        assert m.get("bench_b", 0) > 0

    def test_union_combines_both_modes(self):
        # Two benchmarks, each indexed at a different mode. A document
        # that contains both should match both.
        idx_8 = _make_idx(8, {
            "short_bench": ["alpha beta gamma delta epsilon zeta eta theta"]
        })
        idx_13 = _make_idx(13, {
            "long_bench": [
                "lorem ipsum dolor sit amet consectetur adipiscing elit sed do "
                "eiusmod tempor incididunt ut labore"
            ]
        })
        dual = DualModeDecontaminationIndex(primary=idx_8, secondary=idx_13)
        text = (
            "alpha beta gamma delta epsilon zeta eta theta iota kappa "
            "lorem ipsum dolor sit amet consectetur adipiscing elit sed do "
            "eiusmod tempor incididunt ut labore et dolore magna aliqua"
        )
        m = dual.scan_document(text)
        assert "short_bench" in m, "primary (8-gram) match missing"
        assert "long_bench" in m, "secondary (13-gram) match missing"

    def test_secondary_breakdown_attached_to_report(self):
        # When matches come from the secondary mode, the wrapper attaches
        # a per-benchmark breakdown via report.secondary_per_bench.
        idx_8 = _make_idx(8, {})  # primary has no matches
        idx_13 = _make_idx(13, {
            "ref": [
                "this is a moderately long sentence that fingerprints a "
                "benchmark uniquely across the corpus today"
            ]
        })
        dual = DualModeDecontaminationIndex(primary=idx_8, secondary=idx_13)
        report = DecontaminationReport()
        m = dual.scan_document(
            "this is a moderately long sentence that fingerprints a "
            "benchmark uniquely across the corpus today indeed",
            report=report,
        )
        assert "ref" in m
        # The secondary breakdown is on the report:
        sec = getattr(report, "secondary_per_bench", None)
        assert sec is not None, "secondary_per_bench should be attached to report"
        assert "ref" in sec
        assert sec["ref"]["n_corpus_docs_matched"] == 1

    def test_signatures_property_returns_union(self):
        idx_8 = _make_idx(8, {"bench_a": ["alpha beta gamma delta epsilon zeta eta theta"]})
        idx_13 = _make_idx(13, {
            "bench_b": [
                "lorem ipsum dolor sit amet consectetur adipiscing elit sed do "
                "eiusmod tempor incididunt"
            ]
        })
        dual = DualModeDecontaminationIndex(primary=idx_8, secondary=idx_13)
        sigs = dual.signatures
        # Union of benchmarks across modes
        assert {"bench_a", "bench_b"}.issubset(sigs.keys())
