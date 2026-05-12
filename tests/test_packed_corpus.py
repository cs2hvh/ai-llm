"""Tests for the offline packed-corpus data plane (B2 work).

Pins the file-format contract: anything that breaks round-trip bitwise
equality, the seek index, or the manifest-as-completion-marker
atomicity is a release-blocker for the 1T base run.

What's pinned:
  - Writer → Reader bitwise round-trip across N sequences
  - Seek to arbitrary sequence_id returns correct content
  - Cross-shard reads (sequence_id spans shards)
  - Last-shard partial fill
  - Empty / 1-sequence / many-sequence edge cases
  - Provenance round-trip (DocSpan)
  - Source-mix aggregation in shard manifest
  - uint16-rejection (the 131k-vocab silent-corruption bug)
  - Format version mismatch raises loudly
  - tokenizer_sha256 mismatch raises loudly
  - Atomic write: tokens.bin exists without manifest.json → reader rejects
  - Out-of-range sequence_id raises IndexError
  - Seek-math invariants (shard_id_for, local_offset_for, byte_offset_for)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from myllm.data.packed_corpus import (
    CORPUS_FORMAT_VERSION,
    CorpusManifest,
    DocSpan,
    PackedCorpusReader,
    PackedCorpusWriter,
    SequenceMeta,
    ShardManifest,
    TOKEN_BYTES,
    TOKEN_DTYPE,
    byte_offset_for,
    local_offset_for,
    packed_sequence_bytes,
    shard_id_for,
    write_corpus_manifest,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _make_tokens(sequence_id: int, sequence_length: int) -> np.ndarray:
    """Deterministic token pattern based on sequence_id so we can verify
    bitwise round-trip — every position has a sequence-specific value."""
    base = (sequence_id * 1_000_003) % (2**32)  # large prime
    return np.array(
        [(base + i) % (2**32) for i in range(sequence_length)],
        dtype=TOKEN_DTYPE,
    )


def _make_doc_spans(
    sequence_id: int, sequence_length: int, n_spans: int = 2
) -> list[DocSpan]:
    """Build n_spans contiguous spans covering [0, sequence_length)."""
    span_len = sequence_length // n_spans
    spans = []
    for i in range(n_spans):
        start = i * span_len
        end = (i + 1) * span_len if i < n_spans - 1 else sequence_length
        spans.append(
            DocSpan(
                doc_span_id=-1,  # writer overwrites
                sequence_id=-1,  # writer overwrites
                source_id=("source_a" if i % 2 == 0 else "source_b"),
                doc_id_hash=hash((sequence_id, i)) & ((1 << 63) - 1),
                dataset_revision_id="rev-1",
                token_start_in_sequence=start,
                token_end_in_sequence=end,
                text_hash=(hash(("text", sequence_id, i)) & ((1 << 63) - 1)),
            )
        )
    return spans


def _build_corpus(
    tmp_path: Path,
    *,
    n_sequences: int,
    sequence_length: int,
    sequences_per_shard: int,
    tokenizer_sha256: str = "abc123",
    corpus_name: str = "test",
) -> Path:
    """Write n_sequences and produce a complete corpus (top-level manifest)."""
    root = tmp_path / corpus_name
    w = PackedCorpusWriter(
        root,
        sequence_length=sequence_length,
        sequences_per_shard=sequences_per_shard,
        tokenizer_sha256=tokenizer_sha256,
    )
    for sid in range(n_sequences):
        w.append_sequence(_make_tokens(sid, sequence_length),
                          _make_doc_spans(sid, sequence_length))
    w.close()
    write_corpus_manifest(
        root,
        corpus_name=corpus_name,
        tokenizer_sha256=tokenizer_sha256,
        sequence_length=sequence_length,
        sequences_per_shard=sequences_per_shard,
        source_revisions={"source_a": "rev-1", "source_b": "rev-1"},
        target_source_share={"source_a": 0.5, "source_b": 0.5},
    )
    return root


# --------------------------------------------------------------------------- #
# Seek-math invariants
# --------------------------------------------------------------------------- #
class TestSeekMath:
    def test_first_sequence_in_first_shard(self):
        assert shard_id_for(0, 100) == 0
        assert local_offset_for(0, 100) == 0
        assert byte_offset_for(0, 100, 8) == 0

    def test_within_first_shard(self):
        assert shard_id_for(50, 100) == 0
        assert local_offset_for(50, 100) == 50

    def test_first_sequence_in_second_shard(self):
        assert shard_id_for(100, 100) == 1
        assert local_offset_for(100, 100) == 0
        assert byte_offset_for(100, 100, 8) == 0  # offset within new shard

    def test_byte_offset_uses_uint32_size(self):
        # 4 bytes per token; sequence_length=10 → 40 bytes per sequence.
        assert byte_offset_for(5, 100, 10) == 5 * 10 * 4

    def test_packed_sequence_bytes(self):
        assert packed_sequence_bytes(1024) == 1024 * 4  # uint32

    def test_negative_sequence_id_raises(self):
        with pytest.raises(ValueError):
            shard_id_for(-1, 100)

    def test_zero_sequences_per_shard_raises(self):
        with pytest.raises(ValueError):
            shard_id_for(0, 0)


# --------------------------------------------------------------------------- #
# Round-trip bitwise correctness
# --------------------------------------------------------------------------- #
class TestRoundTrip:
    def test_single_sequence_round_trip(self, tmp_path):
        root = _build_corpus(
            tmp_path, n_sequences=1, sequence_length=16, sequences_per_shard=4,
        )
        r = PackedCorpusReader(root)
        assert r.total_sequences == 1
        assert r.total_tokens == 16
        expected = _make_tokens(0, 16)
        got = r.get_sequence(0)
        np.testing.assert_array_equal(got, expected)
        assert got.dtype == TOKEN_DTYPE

    def test_multi_sequence_within_one_shard(self, tmp_path):
        root = _build_corpus(
            tmp_path, n_sequences=4, sequence_length=16, sequences_per_shard=8,
        )
        r = PackedCorpusReader(root)
        assert r.total_sequences == 4
        for sid in range(4):
            np.testing.assert_array_equal(
                r.get_sequence(sid), _make_tokens(sid, 16),
            )

    def test_sequences_spanning_multiple_shards(self, tmp_path):
        """sequences_per_shard=4, n_sequences=10 → 3 shards (4, 4, 2)."""
        root = _build_corpus(
            tmp_path, n_sequences=10, sequence_length=16, sequences_per_shard=4,
        )
        r = PackedCorpusReader(root)
        assert r.total_sequences == 10
        for sid in range(10):
            np.testing.assert_array_equal(
                r.get_sequence(sid), _make_tokens(sid, 16),
            )

    def test_out_of_order_reads(self, tmp_path):
        """Random-access correctness: reading in reverse must match in-order."""
        root = _build_corpus(
            tmp_path, n_sequences=10, sequence_length=8, sequences_per_shard=3,
        )
        r = PackedCorpusReader(root)
        for sid in [9, 0, 5, 3, 7, 1, 8, 2, 6, 4]:
            np.testing.assert_array_equal(
                r.get_sequence(sid), _make_tokens(sid, 8),
            )

    def test_iterate_from_zero_returns_all(self, tmp_path):
        root = _build_corpus(
            tmp_path, n_sequences=6, sequence_length=8, sequences_per_shard=3,
        )
        r = PackedCorpusReader(root)
        results = list(r.iterate_from(0))
        assert len(results) == 6
        for sid, tokens in results:
            np.testing.assert_array_equal(tokens, _make_tokens(sid, 8))

    def test_iterate_from_offset(self, tmp_path):
        root = _build_corpus(
            tmp_path, n_sequences=6, sequence_length=8, sequences_per_shard=3,
        )
        r = PackedCorpusReader(root)
        results = list(r.iterate_from(4))
        assert [sid for sid, _ in results] == [4, 5]

    def test_last_shard_partial_fill(self, tmp_path):
        """n=10, per_shard=4 → last shard has only 2 sequences. The
        per-shard manifest must reflect this (actual_sequences=2 for shard 2).
        """
        root = _build_corpus(
            tmp_path, n_sequences=10, sequence_length=16, sequences_per_shard=4,
        )
        r = PackedCorpusReader(root)
        sm_last = r.shard_manifest(2)
        assert sm_last.actual_sequences == 2
        assert sm_last.first_sequence_id == 8
        assert sm_last.last_sequence_id == 9
        assert sm_last.total_tokens_in_shard == 2 * 16


# --------------------------------------------------------------------------- #
# uint32 / uint16 dtype safety
# --------------------------------------------------------------------------- #
class TestDtypeSafety:
    def test_uint16_input_coerces_to_uint32_on_disk(self, tmp_path):
        """The writer accepts uint16 input (back-compat with smaller-vocab
        tests) but the on-disk format is uint32."""
        w = PackedCorpusWriter(
            tmp_path / "c", sequence_length=8, sequences_per_shard=2,
            tokenizer_sha256="x",
        )
        u16 = np.arange(8, dtype=np.uint16)
        w.append_sequence(u16, _make_doc_spans(0, 8))
        w.close()
        # On-disk file size must be 8 * 4 = 32 bytes, not 8 * 2 = 16.
        token_path = tmp_path / "c" / "shard-000000" / "tokens.bin"
        assert token_path.stat().st_size == 8 * TOKEN_BYTES

    def test_oversized_token_id_fits_uint32(self, tmp_path):
        """A token id 131,071 (our SPM-Unigram vocab top) MUST round-trip."""
        w = PackedCorpusWriter(
            tmp_path / "c", sequence_length=4, sequences_per_shard=1,
            tokenizer_sha256="x",
        )
        tokens = np.array([131_070, 131_071, 65_535, 0], dtype=TOKEN_DTYPE)
        w.append_sequence(tokens, _make_doc_spans(0, 4))
        w.close()
        write_corpus_manifest(
            tmp_path / "c",
            corpus_name="c", tokenizer_sha256="x",
            sequence_length=4, sequences_per_shard=1,
            source_revisions={"source_a": "rev-1", "source_b": "rev-1"},
            target_source_share={"source_a": 0.5, "source_b": 0.5},
        )
        r = PackedCorpusReader(tmp_path / "c")
        got = r.get_sequence(0)
        # Critical: 131,071 round-trips. With uint16 it would wrap to a
        # different value. This is the silent-corruption bug class.
        assert got[1] == 131_071
        assert got[0] == 131_070

    def test_negative_signed_input_rejected(self, tmp_path):
        """Negative signed ints can't fit in uint32 — reject loudly."""
        w = PackedCorpusWriter(
            tmp_path / "c", sequence_length=4, sequences_per_shard=1,
            tokenizer_sha256="x",
        )
        bad = np.array([1, 2, -3, 4], dtype=np.int32)
        with pytest.raises(ValueError, match="negative"):
            w.append_sequence(bad, _make_doc_spans(0, 4))


# --------------------------------------------------------------------------- #
# Validation: shape, length, doc-span bounds
# --------------------------------------------------------------------------- #
class TestValidation:
    def test_wrong_sequence_length_raises(self, tmp_path):
        w = PackedCorpusWriter(
            tmp_path / "c", sequence_length=8, sequences_per_shard=2,
            tokenizer_sha256="x",
        )
        with pytest.raises(ValueError, match="length"):
            w.append_sequence(np.arange(7, dtype=TOKEN_DTYPE),
                              _make_doc_spans(0, 8))

    def test_2d_input_rejected(self, tmp_path):
        w = PackedCorpusWriter(
            tmp_path / "c", sequence_length=8, sequences_per_shard=2,
            tokenizer_sha256="x",
        )
        with pytest.raises(ValueError, match="1-D"):
            w.append_sequence(
                np.zeros((2, 4), dtype=TOKEN_DTYPE), _make_doc_spans(0, 8),
            )

    def test_doc_span_past_sequence_length_rejected(self, tmp_path):
        w = PackedCorpusWriter(
            tmp_path / "c", sequence_length=8, sequences_per_shard=2,
            tokenizer_sha256="x",
        )
        bad_span = DocSpan(
            doc_span_id=-1, sequence_id=-1,
            source_id="x", doc_id_hash=1, dataset_revision_id="r",
            token_start_in_sequence=0, token_end_in_sequence=99,  # past end
            text_hash=1,
        )
        with pytest.raises(ValueError, match="sequence_length"):
            w.append_sequence(np.zeros(8, dtype=TOKEN_DTYPE), [bad_span])

    def test_doc_span_negative_start_rejected(self, tmp_path):
        w = PackedCorpusWriter(
            tmp_path / "c", sequence_length=8, sequences_per_shard=2,
            tokenizer_sha256="x",
        )
        bad_span = DocSpan(
            doc_span_id=-1, sequence_id=-1,
            source_id="x", doc_id_hash=1, dataset_revision_id="r",
            token_start_in_sequence=-1, token_end_in_sequence=4,
            text_hash=1,
        )
        with pytest.raises(ValueError, match="< 0"):
            w.append_sequence(np.zeros(8, dtype=TOKEN_DTYPE), [bad_span])


# --------------------------------------------------------------------------- #
# Provenance round-trip
# --------------------------------------------------------------------------- #
class TestProvenance:
    def test_get_provenance_returns_assigned_spans(self, tmp_path):
        root = _build_corpus(
            tmp_path, n_sequences=4, sequence_length=16, sequences_per_shard=4,
        )
        r = PackedCorpusReader(root)
        spans = r.get_provenance(sequence_id=2)
        assert len(spans) == 2  # _make_doc_spans uses n_spans=2 default
        assert all(s.sequence_id == 2 for s in spans)
        # doc_span_ids should be unique + assigned by writer (not the -1 sentinel).
        ids = [s.doc_span_id for s in spans]
        assert -1 not in ids
        assert len(set(ids)) == len(ids)

    def test_provenance_doc_span_ids_are_contiguous_across_corpus(self, tmp_path):
        """doc_span_id must be unique + monotone across the corpus —
        the launcher uses this for global cross-shard ordering."""
        root = _build_corpus(
            tmp_path, n_sequences=6, sequence_length=8, sequences_per_shard=3,
        )
        r = PackedCorpusReader(root)
        all_ids: list[int] = []
        for sid in range(6):
            all_ids.extend(s.doc_span_id for s in r.get_provenance(sid))
        # 6 sequences × 2 spans each = 12 spans, ids 0..11
        assert all_ids == list(range(12))

    def test_source_mix_in_sequence_meta(self, tmp_path):
        root = _build_corpus(
            tmp_path, n_sequences=2, sequence_length=8, sequences_per_shard=2,
        )
        r = PackedCorpusReader(root)
        meta = r.get_sequence_meta(0)
        # _make_doc_spans creates one source_a span (positions 0..4) +
        # one source_b span (positions 4..8) — 4 tokens each.
        assert meta.source_mix == {"source_a": 4, "source_b": 4}

    def test_shard_source_mix_aggregates(self, tmp_path):
        root = _build_corpus(
            tmp_path, n_sequences=3, sequence_length=10, sequences_per_shard=3,
        )
        r = PackedCorpusReader(root)
        sm = r.shard_manifest(0)
        # 3 sequences × 5 tokens each per source.
        # _make_doc_spans n_spans=2: span 0 = [0, 5), span 1 = [5, 10) → 5 each
        assert sm.source_mix_histogram == {"source_a": 15, "source_b": 15}


# --------------------------------------------------------------------------- #
# Manifest atomicity — the "manifest is the completion marker" contract
# --------------------------------------------------------------------------- #
class TestManifestAtomicity:
    def test_missing_top_level_manifest_raises(self, tmp_path):
        w = PackedCorpusWriter(
            tmp_path / "c", sequence_length=8, sequences_per_shard=2,
            tokenizer_sha256="x",
        )
        w.append_sequence(np.zeros(8, dtype=TOKEN_DTYPE), _make_doc_spans(0, 8))
        w.close()
        # Did NOT call write_corpus_manifest — top-level manifest is missing.
        with pytest.raises(FileNotFoundError, match="manifest"):
            PackedCorpusReader(tmp_path / "c")

    def test_format_version_mismatch_raises(self, tmp_path):
        root = _build_corpus(
            tmp_path, n_sequences=1, sequence_length=4, sequences_per_shard=1,
        )
        # Tamper with the manifest's format_version.
        mp = root / "manifest.json"
        d = json.loads(mp.read_text())
        d["format_version"] = CORPUS_FORMAT_VERSION + 99
        mp.write_text(json.dumps(d))
        with pytest.raises(ValueError, match="format_version"):
            PackedCorpusReader(root)

    def test_tokenizer_sha_mismatch_in_aggregate_raises(self, tmp_path):
        """If a shard's tokenizer_sha256 differs from the corpus-level
        value, write_corpus_manifest must refuse (silently mixed
        tokenizers would corrupt training)."""
        root = tmp_path / "c"
        w = PackedCorpusWriter(
            root, sequence_length=4, sequences_per_shard=1,
            tokenizer_sha256="DIFFERENT",
        )
        w.append_sequence(np.zeros(4, dtype=TOKEN_DTYPE), _make_doc_spans(0, 4))
        w.close()
        with pytest.raises(ValueError, match="tokenizer_sha256"):
            write_corpus_manifest(
                root,
                corpus_name="c", tokenizer_sha256="EXPECTED",
                sequence_length=4, sequences_per_shard=1,
                source_revisions={"source_a": "r"},
                target_source_share={"source_a": 1.0},
            )

    def test_sequence_length_mismatch_in_aggregate_raises(self, tmp_path):
        root = tmp_path / "c"
        w = PackedCorpusWriter(
            root, sequence_length=4, sequences_per_shard=1,
            tokenizer_sha256="x",
        )
        w.append_sequence(np.zeros(4, dtype=TOKEN_DTYPE), _make_doc_spans(0, 4))
        w.close()
        with pytest.raises(ValueError, match="sequence_length"):
            write_corpus_manifest(
                root,
                corpus_name="c", tokenizer_sha256="x",
                sequence_length=99,
                sequences_per_shard=1,
                source_revisions={"source_a": "r"},
                target_source_share={"source_a": 1.0},
            )


# --------------------------------------------------------------------------- #
# Reader out-of-range + LRU behaviour
# --------------------------------------------------------------------------- #
class TestReaderEdgeCases:
    def test_out_of_range_sequence_id_raises(self, tmp_path):
        root = _build_corpus(
            tmp_path, n_sequences=4, sequence_length=8, sequences_per_shard=2,
        )
        r = PackedCorpusReader(root)
        with pytest.raises(IndexError):
            r.get_sequence(4)
        with pytest.raises(IndexError):
            r.get_sequence(-1)

    def test_iterate_negative_start_raises(self, tmp_path):
        root = _build_corpus(
            tmp_path, n_sequences=4, sequence_length=8, sequences_per_shard=2,
        )
        r = PackedCorpusReader(root)
        with pytest.raises(ValueError):
            list(r.iterate_from(-1))

    def test_lru_evicts_oldest_shard(self, tmp_path):
        """max_open_shards=2; touch 3 shards → first should be evicted."""
        root = _build_corpus(
            tmp_path, n_sequences=9, sequence_length=8, sequences_per_shard=3,
        )
        r = PackedCorpusReader(root, max_open_shards=2)
        r.get_sequence(0)  # opens shard 0
        r.get_sequence(3)  # opens shard 1
        r.get_sequence(6)  # opens shard 2 → evicts shard 0
        assert 0 not in r._open_token_maps
        assert 1 in r._open_token_maps
        assert 2 in r._open_token_maps


# --------------------------------------------------------------------------- #
# Actual share computation
# --------------------------------------------------------------------------- #
class TestActualShareComputation:
    def test_actual_share_sums_to_one(self, tmp_path):
        root = _build_corpus(
            tmp_path, n_sequences=5, sequence_length=8, sequences_per_shard=2,
        )
        r = PackedCorpusReader(root)
        m = r.manifest
        total = sum(m.actual_source_share.values())
        assert abs(total - 1.0) < 1e-9

    def test_actual_share_matches_50_50_pattern(self, tmp_path):
        """_make_doc_spans splits each sequence 50/50 between source_a and
        source_b — actual_source_share should be {a: 0.5, b: 0.5}."""
        root = _build_corpus(
            tmp_path, n_sequences=4, sequence_length=8, sequences_per_shard=2,
        )
        r = PackedCorpusReader(root)
        assert abs(r.manifest.actual_source_share["source_a"] - 0.5) < 1e-9
        assert abs(r.manifest.actual_source_share["source_b"] - 0.5) < 1e-9
