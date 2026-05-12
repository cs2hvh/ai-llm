"""Tests for cross-source compose pass.

Pins the token-share-enforcement contract: with target shares
{src_a: 0.7, src_b: 0.3}, the output corpus's actual_source_share
should converge to (0.7, 0.3) ± noise floor in the long run.

Other pinned contracts:
  - Tokenizer SHA mismatch across sources → raise loudly
  - Sequence length mismatch → raise
  - target_shares must sum to 1 (validation)
  - Per-source revisions propagated into output manifest
  - Exhausted sources detected
  - Provenance preserved end-to-end (DocSpan source_id, revision_id)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from myllm.data.compose import ComposeStats, compose_mixed_corpus
from myllm.data.packed_corpus import (
    DocSpan,
    PackedCorpusReader,
    PackedCorpusWriter,
    TOKEN_DTYPE,
    write_corpus_manifest,
)


# --------------------------------------------------------------------------- #
# Helpers — build minimal per-source corpora to feed compose
# --------------------------------------------------------------------------- #
def _build_synthetic_source(
    root: Path,
    *,
    source_id: str,
    n_sequences: int,
    sequence_length: int = 8,
    sequences_per_shard: int = 4,
    tokenizer_sha256: str = "TOK",
    revision_id: str = "rev-1",
) -> Path:
    """Build a tiny per-source corpus with synthetic tokens + provenance."""
    w = PackedCorpusWriter(
        root,
        sequence_length=sequence_length,
        sequences_per_shard=sequences_per_shard,
        tokenizer_sha256=tokenizer_sha256,
    )
    # Mark each sequence's tokens with a source-distinguishable pattern.
    seed = abs(hash(source_id)) % 1_000_000
    for sid in range(n_sequences):
        tokens = np.array(
            [(seed + sid * 1000 + i) % (1 << 30) for i in range(sequence_length)],
            dtype=TOKEN_DTYPE,
        )
        spans = [
            DocSpan(
                doc_span_id=-1, sequence_id=-1,
                source_id=source_id,
                doc_id_hash=(abs(hash((source_id, sid))) % (1 << 63)),
                dataset_revision_id=revision_id,
                token_start_in_sequence=0,
                token_end_in_sequence=sequence_length,
                text_hash=(abs(hash((source_id, "text", sid))) % (1 << 63)),
            ),
        ]
        w.append_sequence(tokens, spans)
    w.close()
    write_corpus_manifest(
        root,
        corpus_name=source_id,
        tokenizer_sha256=tokenizer_sha256,
        sequence_length=sequence_length,
        sequences_per_shard=sequences_per_shard,
        source_revisions={source_id: revision_id},
        target_source_share={source_id: 1.0},
    )
    return root


# --------------------------------------------------------------------------- #
# Validation tests
# --------------------------------------------------------------------------- #
class TestValidation:
    def test_empty_sources_raises(self, tmp_path):
        with pytest.raises(ValueError, match="empty"):
            compose_mixed_corpus(
                {}, tmp_path / "out",
                target_shares={}, sequences_per_shard=2,
            )

    def test_target_shares_must_sum_to_one(self, tmp_path):
        a = _build_synthetic_source(tmp_path / "a", source_id="a", n_sequences=1)
        b = _build_synthetic_source(tmp_path / "b", source_id="b", n_sequences=1)
        with pytest.raises(ValueError, match="sum to 1"):
            compose_mixed_corpus(
                {"a": a, "b": b}, tmp_path / "out",
                target_shares={"a": 0.5, "b": 0.6},  # sum=1.1
                sequences_per_shard=2,
            )

    def test_missing_target_share_raises(self, tmp_path):
        a = _build_synthetic_source(tmp_path / "a", source_id="a", n_sequences=1)
        b = _build_synthetic_source(tmp_path / "b", source_id="b", n_sequences=1)
        with pytest.raises(ValueError, match="missing"):
            compose_mixed_corpus(
                {"a": a, "b": b}, tmp_path / "out",
                target_shares={"a": 1.0},  # missing b
                sequences_per_shard=2,
            )

    def test_extra_target_share_raises(self, tmp_path):
        a = _build_synthetic_source(tmp_path / "a", source_id="a", n_sequences=1)
        with pytest.raises(ValueError, match="no matching"):
            compose_mixed_corpus(
                {"a": a}, tmp_path / "out",
                target_shares={"a": 0.5, "b": 0.5},  # b doesn't exist
                sequences_per_shard=2,
            )

    def test_tokenizer_sha_mismatch_raises(self, tmp_path):
        a = _build_synthetic_source(
            tmp_path / "a", source_id="a", n_sequences=2, tokenizer_sha256="AAA",
        )
        b = _build_synthetic_source(
            tmp_path / "b", source_id="b", n_sequences=2, tokenizer_sha256="BBB",
        )
        with pytest.raises(ValueError, match="tokenizer_sha256"):
            compose_mixed_corpus(
                {"a": a, "b": b}, tmp_path / "out",
                target_shares={"a": 0.5, "b": 0.5},
                sequences_per_shard=2,
            )

    def test_sequence_length_mismatch_raises(self, tmp_path):
        a = _build_synthetic_source(
            tmp_path / "a", source_id="a", n_sequences=2, sequence_length=8,
        )
        b = _build_synthetic_source(
            tmp_path / "b", source_id="b", n_sequences=2, sequence_length=16,
        )
        with pytest.raises(ValueError, match="sequence_length"):
            compose_mixed_corpus(
                {"a": a, "b": b}, tmp_path / "out",
                target_shares={"a": 0.5, "b": 0.5},
                sequences_per_shard=2,
            )


# --------------------------------------------------------------------------- #
# Token-share enforcement
# --------------------------------------------------------------------------- #
class TestTokenShareEnforcement:
    def test_50_50_split_converges(self, tmp_path):
        a = _build_synthetic_source(tmp_path / "a", source_id="a", n_sequences=50)
        b = _build_synthetic_source(tmp_path / "b", source_id="b", n_sequences=50)
        stats, manifest = compose_mixed_corpus(
            {"a": a, "b": b}, tmp_path / "out",
            target_shares={"a": 0.5, "b": 0.5},
            sequences_per_shard=8,
            target_total_sequences=80,
        )
        assert stats.total_sequences_emitted == 80
        # 50/50 target with discrete sequences → 40 each ±1
        assert abs(stats.per_source_sequences["a"] - 40) <= 1
        assert abs(stats.per_source_sequences["b"] - 40) <= 1
        # Manifest actual_source_share should also be close to 0.5/0.5
        assert abs(manifest.actual_source_share["a"] - 0.5) < 0.05
        assert abs(manifest.actual_source_share["b"] - 0.5) < 0.05

    def test_70_30_split_converges(self, tmp_path):
        a = _build_synthetic_source(tmp_path / "a", source_id="a", n_sequences=200)
        b = _build_synthetic_source(tmp_path / "b", source_id="b", n_sequences=200)
        stats, _ = compose_mixed_corpus(
            {"a": a, "b": b}, tmp_path / "out",
            target_shares={"a": 0.7, "b": 0.3},
            sequences_per_shard=16,
            target_total_sequences=200,
        )
        # 70/30 of 200 = 140/60, tolerance ±2 from rounding + bootstrap.
        assert abs(stats.per_source_sequences["a"] - 140) <= 3
        assert abs(stats.per_source_sequences["b"] - 60) <= 3

    def test_three_source_unequal_split(self, tmp_path):
        a = _build_synthetic_source(tmp_path / "a", source_id="a", n_sequences=100)
        b = _build_synthetic_source(tmp_path / "b", source_id="b", n_sequences=100)
        c = _build_synthetic_source(tmp_path / "c", source_id="c", n_sequences=100)
        stats, _ = compose_mixed_corpus(
            {"a": a, "b": b, "c": c}, tmp_path / "out",
            target_shares={"a": 0.6, "b": 0.3, "c": 0.1},
            sequences_per_shard=16,
            target_total_sequences=150,
        )
        # Of 150 sequences: ~90/~45/~15 by target shares.
        assert abs(stats.per_source_sequences["a"] - 90) <= 3
        assert abs(stats.per_source_sequences["b"] - 45) <= 3
        assert abs(stats.per_source_sequences["c"] - 15) <= 3


# --------------------------------------------------------------------------- #
# Exhaustion + provenance + round-trip
# --------------------------------------------------------------------------- #
class TestExhaustionAndProvenance:
    def test_exhausted_source_detected(self, tmp_path):
        """Source 'a' has 10 sequences; with 50/50 target after 20 picks
        we'd want 10 from each — 'a' exhausts exactly. After 'a' runs out,
        compose continues with only 'b'."""
        a = _build_synthetic_source(tmp_path / "a", source_id="a", n_sequences=10)
        b = _build_synthetic_source(tmp_path / "b", source_id="b", n_sequences=50)
        stats, _ = compose_mixed_corpus(
            {"a": a, "b": b}, tmp_path / "out",
            target_shares={"a": 0.5, "b": 0.5},
            sequences_per_shard=8,
            target_total_sequences=30,
        )
        assert "a" in stats.exhausted_sources
        # Source a contributed at most its size of 10.
        assert stats.per_source_sequences["a"] <= 10
        # Remaining 20+ came from b.
        assert stats.per_source_sequences["b"] >= 20

    def test_provenance_source_id_preserved(self, tmp_path):
        a = _build_synthetic_source(tmp_path / "a", source_id="src_aaa", n_sequences=10)
        b = _build_synthetic_source(tmp_path / "b", source_id="src_bbb", n_sequences=10)
        _, _ = compose_mixed_corpus(
            {"src_aaa": a, "src_bbb": b}, tmp_path / "out",
            target_shares={"src_aaa": 0.5, "src_bbb": 0.5},
            sequences_per_shard=4,
            target_total_sequences=12,
        )
        r = PackedCorpusReader(tmp_path / "out")
        seen_sources = set()
        for sid in range(r.total_sequences):
            for span in r.get_provenance(sid):
                seen_sources.add(span.source_id)
        assert seen_sources == {"src_aaa", "src_bbb"}

    def test_revision_id_in_output_manifest(self, tmp_path):
        a = _build_synthetic_source(
            tmp_path / "a", source_id="a", n_sequences=5, revision_id="commit-aaa",
        )
        b = _build_synthetic_source(
            tmp_path / "b", source_id="b", n_sequences=5, revision_id="commit-bbb",
        )
        _, manifest = compose_mixed_corpus(
            {"a": a, "b": b}, tmp_path / "out",
            target_shares={"a": 0.5, "b": 0.5},
            sequences_per_shard=4,
            target_total_sequences=8,
        )
        assert manifest.source_revisions == {"a": "commit-aaa", "b": "commit-bbb"}

    def test_round_trip_tokens_match_source(self, tmp_path):
        """Tokens emitted by compose must match exactly what the source
        reader returns at the consumed local sequence_id."""
        a = _build_synthetic_source(
            tmp_path / "a", source_id="a", n_sequences=5,
            sequence_length=8, sequences_per_shard=5,
        )
        # Single source means every output sequence comes from 'a' in order.
        stats, _ = compose_mixed_corpus(
            {"a": a}, tmp_path / "out",
            target_shares={"a": 1.0},
            sequences_per_shard=5,
            target_total_sequences=5,
        )
        assert stats.total_sequences_emitted == 5
        a_reader = PackedCorpusReader(a)
        out_reader = PackedCorpusReader(tmp_path / "out")
        for sid in range(5):
            np.testing.assert_array_equal(
                out_reader.get_sequence(sid),
                a_reader.get_sequence(sid),
            )


# --------------------------------------------------------------------------- #
# ComposeStats output structure
# --------------------------------------------------------------------------- #
class TestComposeStats:
    def test_per_source_tokens_equals_sequences_times_seq_len(self, tmp_path):
        a = _build_synthetic_source(
            tmp_path / "a", source_id="a", n_sequences=10, sequence_length=8,
        )
        b = _build_synthetic_source(
            tmp_path / "b", source_id="b", n_sequences=10, sequence_length=8,
        )
        stats, _ = compose_mixed_corpus(
            {"a": a, "b": b}, tmp_path / "out",
            target_shares={"a": 0.5, "b": 0.5},
            sequences_per_shard=4,
            target_total_sequences=12,
        )
        for sid, n_seqs in stats.per_source_sequences.items():
            assert stats.per_source_tokens[sid] == n_seqs * 8

    def test_total_tokens_consistent(self, tmp_path):
        a = _build_synthetic_source(
            tmp_path / "a", source_id="a", n_sequences=5, sequence_length=8,
        )
        stats, _ = compose_mixed_corpus(
            {"a": a}, tmp_path / "out",
            target_shares={"a": 1.0},
            sequences_per_shard=2,
            target_total_sequences=5,
        )
        assert stats.total_tokens_emitted == 5 * 8
