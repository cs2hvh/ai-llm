"""End-to-end tests for the build pipeline (build_one_source +
pack_with_provenance) using synthetic documents.

These don't touch HF or a real tokenizer — they inject a stub
tokenizer that produces deterministic token streams, and feed
hand-built Document objects through build_one_source. The output is
then read back via PackedCorpusReader and verified for:

  - Token round-trip
  - Provenance preservation (source_id, revision_id, doc_id_hash)
  - Dedupe correctly filters near-duplicates
  - Filter callback correctly removes docs
  - Decontaminator correctly removes contaminated docs
  - Cross-sequence document spans produce TWO DocSpan rows
  - sample_limit caps emission
  - BuildStats counters are accurate
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from myllm.data.build import (
    BuildStats,
    FINEWEB_MINHASH_CONFIG,
    _text_hash,
    build_one_source,
    pack_with_provenance,
)
from myllm.data.dedupe import MinHashConfig
from myllm.data.packed_corpus import (
    DocSpan,
    PackedCorpusReader,
    TOKEN_DTYPE,
)
from myllm.data.types import Document


# --------------------------------------------------------------------------- #
# Stub tokenizer + helpers
# --------------------------------------------------------------------------- #
class _StubTokenizer:
    """Deterministic tokenizer: each word → its position * 17 + 100 (mod 2^20).

    Reasonable scale (tokens land in [100, ~1M]) so 131k-vocab edge cases
    aren't exercised — but uint32-safe.
    """

    def encode(self, text: str):
        words = text.split()
        ids = [((i * 17 + 100 + hash(w) & 0xFFFF) % 1_000_000) for i, w in enumerate(words)]
        # Inner result must look like an HF tokenizers Encoding (only .ids used).
        return _StubEncoding(ids)


class _StubEncoding:
    def __init__(self, ids: list[int]):
        self.ids = ids


def _make_doc(doc_id: str, text: str, source: str = "web") -> Document:
    return Document(
        text=text,
        source=source,  # type: ignore[arg-type]
        dataset="test/synth",
        doc_id=doc_id,
    )


# --------------------------------------------------------------------------- #
# pack_with_provenance — unit-level
# --------------------------------------------------------------------------- #
class TestPackWithProvenance:
    def test_single_doc_within_one_sequence(self):
        """Doc of 6 tokens + EOS = 7 tokens; sequence_length=8 means doc
        fits in one sequence (with padding NOT applied since drop_last=True
        and we have nothing left)."""
        docs = [("d1", "src", [1, 2, 3, 4, 5, 6], 42, "rev-1")]
        # Drop the trailing partial since 7 < 8.
        results = list(pack_with_provenance(
            docs, sequence_length=8, eos_token_id=99, drop_last=True,
        ))
        # 7 tokens < 8, so nothing emitted.
        assert results == []

    def test_drop_last_false_emits_padded_sequence(self):
        """With drop_last=False, the trailing 7-token doc + EOS gets padded."""
        docs = [("d1", "src", [1, 2, 3, 4, 5, 6], 42, "rev-1")]
        results = list(pack_with_provenance(
            docs, sequence_length=8, eos_token_id=99, pad_token_id=0, drop_last=False,
        ))
        assert len(results) == 1
        tokens, spans = results[0]
        # 6 content tokens + EOS (99) + 1 pad (0)
        np.testing.assert_array_equal(tokens, [1, 2, 3, 4, 5, 6, 99, 0])
        assert len(spans) == 1
        assert spans[0].source_id == "src"
        assert spans[0].token_start_in_sequence == 0
        assert spans[0].token_end_in_sequence == 7  # excludes the pad byte
        assert spans[0].dataset_revision_id == "rev-1"
        assert spans[0].text_hash == 42

    def test_doc_spans_sequence_boundary_produces_two_spans(self):
        """A doc that crosses a sequence boundary must produce one DocSpan
        in each sequence — per-sequence provenance must be complete."""
        # One doc of 12 tokens + EOS = 13 tokens; seq_len=8.
        # First sequence: tokens 0..7 of the doc.
        # Second sequence: tokens 8..11 of the doc + EOS + maybe more from next doc.
        docs = [
            ("d1", "src_a", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 100, "rev-1"),
            ("d2", "src_a", [50, 51, 52, 53], 200, "rev-1"),
        ]
        results = list(pack_with_provenance(
            docs, sequence_length=8, eos_token_id=99, drop_last=True,
        ))
        # 13 + 5 = 18 tokens → 2 full sequences of 8, 2 left → dropped.
        assert len(results) == 2

        seq1_tokens, seq1_spans = results[0]
        np.testing.assert_array_equal(seq1_tokens, [1, 2, 3, 4, 5, 6, 7, 8])
        # Doc 1 occupies positions 0..7 of seq 1 — one span.
        assert len(seq1_spans) == 1
        assert seq1_spans[0].text_hash == 100
        assert seq1_spans[0].token_start_in_sequence == 0
        assert seq1_spans[0].token_end_in_sequence == 8

        seq2_tokens, seq2_spans = results[1]
        # Sequence 2: tokens 9..12 of doc 1 (4 tokens) + EOS (99) + tokens 50,51,52 of doc 2 (3 tokens).
        np.testing.assert_array_equal(seq2_tokens, [9, 10, 11, 12, 99, 50, 51, 52])
        # Two spans: doc 1's tail (positions 0..5 including EOS), doc 2's head (positions 5..8).
        assert len(seq2_spans) == 2
        # Doc 1 carry-over span:
        carry = next(s for s in seq2_spans if s.text_hash == 100)
        assert carry.token_start_in_sequence == 0
        assert carry.token_end_in_sequence == 5  # 4 content + 1 EOS
        # Doc 2 head span:
        head = next(s for s in seq2_spans if s.text_hash == 200)
        assert head.token_start_in_sequence == 5
        assert head.token_end_in_sequence == 8

    def test_empty_doc_skipped(self):
        docs = [
            ("d1", "src", [], 1, "r"),
            ("d2", "src", [1, 2, 3, 4, 5, 6, 7], 2, "r"),
        ]
        results = list(pack_with_provenance(
            docs, sequence_length=8, eos_token_id=99, drop_last=False,
        ))
        assert len(results) == 1
        _, spans = results[0]
        # Only doc 2 contributed.
        assert all(s.text_hash == 2 for s in spans)

    def test_doc_id_hash_is_stable(self):
        """Same doc_id → same doc_id_hash. Tests deterministic xxhash mapping."""
        docs1 = [("doc-abc", "src", [1, 2, 3, 4, 5, 6, 7], 1, "r")]
        docs2 = [("doc-abc", "src", [1, 2, 3, 4, 5, 6, 7], 1, "r")]
        r1 = list(pack_with_provenance(docs1, sequence_length=8, eos_token_id=99, drop_last=False))
        r2 = list(pack_with_provenance(docs2, sequence_length=8, eos_token_id=99, drop_last=False))
        assert r1[0][1][0].doc_id_hash == r2[0][1][0].doc_id_hash

    def test_token_dtype_is_uint32(self):
        docs = [("d1", "src", [131_070, 131_071, 5, 6, 7, 8, 9], 1, "r")]
        results = list(pack_with_provenance(
            docs, sequence_length=8, eos_token_id=99, drop_last=False,
        ))
        tokens, _ = results[0]
        assert tokens.dtype == TOKEN_DTYPE
        # 131_071 must round-trip (uint16 would wrap).
        assert int(tokens[1]) == 131_071


# --------------------------------------------------------------------------- #
# build_one_source — end-to-end with stub tokenizer
# --------------------------------------------------------------------------- #
class TestBuildOneSource:
    def test_round_trip_with_synthetic_docs(self, tmp_path):
        docs = [_make_doc(f"d{i}", f"document number {i} has several words " * 3)
                for i in range(10)]
        tok = _StubTokenizer()
        stats, manifest = build_one_source(
            source_id="synth",
            docs=docs,
            tokenizer=tok,
            output_dir=tmp_path / "synth_corpus",
            sequence_length=32,
            sequences_per_shard=4,
            revision_id="rev-synth-2026-05-12",
            tokenizer_sha256="deadbeef",
            eos_token_id=1,
            dedupe_config=None,  # no dedupe for clarity
            drop_last=False,
        )
        assert stats.docs_seen == 10
        assert stats.docs_kept == 10
        assert stats.sequences_emitted >= 1

        # Read back and verify provenance.
        r = PackedCorpusReader(tmp_path / "synth_corpus")
        assert r.total_sequences == stats.sequences_emitted
        # First sequence must have provenance pointing at our synth source.
        spans = r.get_provenance(0)
        assert len(spans) >= 1
        assert spans[0].source_id == "synth"
        assert spans[0].dataset_revision_id == "rev-synth-2026-05-12"

    def test_filter_fn_drops_short_docs(self, tmp_path):
        docs = [
            _make_doc("d1", "tiny"),                                       # 1 word
            _make_doc("d2", "long enough document with many words here " * 5),
        ]
        def keep_long(doc):
            return len(doc.text.split()) >= 10

        stats, _ = build_one_source(
            source_id="synth",
            docs=docs,
            tokenizer=_StubTokenizer(),
            output_dir=tmp_path / "c",
            sequence_length=32,
            sequences_per_shard=2,
            revision_id="r",
            tokenizer_sha256="x",
            eos_token_id=1,
            filter_fn=keep_long,
            dedupe_config=None,
            drop_last=False,
        )
        assert stats.docs_seen == 2
        assert stats.docs_filtered == 1
        assert stats.docs_kept == 1

    def test_dedupe_removes_near_duplicates(self, tmp_path):
        """Identical docs → dedupe filters all but the first."""
        text = "this is a very long document with many many many repeated words " * 10
        docs = [_make_doc(f"d{i}", text) for i in range(5)]
        stats, _ = build_one_source(
            source_id="synth",
            docs=docs,
            tokenizer=_StubTokenizer(),
            output_dir=tmp_path / "c",
            sequence_length=32,
            sequences_per_shard=2,
            revision_id="r",
            tokenizer_sha256="x",
            eos_token_id=1,
            dedupe_config=FINEWEB_MINHASH_CONFIG,
            drop_last=False,
        )
        assert stats.docs_seen == 5
        assert stats.docs_kept == 1
        assert stats.docs_deduped == 4

    def test_decontaminator_removes_contaminated_docs(self, tmp_path):
        """Passing a decontaminator with a positive scan_document result
        removes the doc and increments docs_contaminated."""

        class _StubDecontaminator:
            def __init__(self, contaminated_ids: set[str]):
                self.contaminated_ids = contaminated_ids

            def scan_document(self, text):
                # Mark text containing "TAINTED" as contaminated.
                return {"benchmark": 1} if "TAINTED" in text else {}

        docs = [
            _make_doc("d1", "clean document text TAINTED by benchmark"),
            _make_doc("d2", "perfectly clean document with no issues"),
        ]
        stats, _ = build_one_source(
            source_id="synth",
            docs=docs,
            tokenizer=_StubTokenizer(),
            output_dir=tmp_path / "c",
            sequence_length=16,
            sequences_per_shard=2,
            revision_id="r",
            tokenizer_sha256="x",
            eos_token_id=1,
            decontaminator=_StubDecontaminator({"d1"}),
            dedupe_config=None,
            drop_last=False,
        )
        assert stats.docs_contaminated == 1
        assert stats.docs_kept == 1

    def test_sample_limit_caps_emission(self, tmp_path):
        docs = [_make_doc(f"d{i}", f"document {i} text words words words " * 5)
                for i in range(20)]
        stats, _ = build_one_source(
            source_id="synth",
            docs=docs,
            tokenizer=_StubTokenizer(),
            output_dir=tmp_path / "c",
            sequence_length=32,
            sequences_per_shard=2,
            revision_id="r",
            tokenizer_sha256="x",
            eos_token_id=1,
            sample_limit=5,
            dedupe_config=None,
            drop_last=False,
        )
        assert stats.docs_kept == 5

    def test_manifest_revision_id_recorded(self, tmp_path):
        docs = [_make_doc("d1", "some text " * 10)]
        _, manifest = build_one_source(
            source_id="synth",
            docs=docs,
            tokenizer=_StubTokenizer(),
            output_dir=tmp_path / "c",
            sequence_length=16,
            sequences_per_shard=2,
            revision_id="abc123commitsha",
            tokenizer_sha256="x",
            eos_token_id=1,
            dedupe_config=None,
            drop_last=False,
        )
        assert manifest.source_revisions == {"synth": "abc123commitsha"}

    def test_target_share_in_manifest_is_within_corpus(self, tmp_path):
        """Per-source corpus's MANIFEST target_source_share = {src: 1.0}
        always (within-corpus, the corpus is 100% one source). The
        production-mix target share is informational and lives in
        BuildStats.production_target_share."""
        docs = [_make_doc("d1", "some text " * 10)]
        stats, manifest = build_one_source(
            source_id="src_b",
            docs=docs,
            tokenizer=_StubTokenizer(),
            output_dir=tmp_path / "c",
            sequence_length=16,
            sequences_per_shard=2,
            revision_id="r",
            tokenizer_sha256="x",
            eos_token_id=1,
            dedupe_config=None,
            target_share=0.18,
            drop_last=False,
        )
        # WITHIN-corpus target = 1.0 (single-source corpus).
        assert manifest.target_source_share == {"src_b": 1.0}
        # Production-mix target preserved on stats for the launcher.
        assert stats.production_target_share == 0.18

    def test_stats_counters_sum_correctly(self, tmp_path):
        """docs_seen == docs_kept + docs_filtered + docs_contaminated + docs_deduped."""

        class _StubDecon:
            def scan_document(self, text):
                return {"bench": 1} if "BAD" in text else {}

        docs = [
            _make_doc("d1", "short"),                                         # filtered
            _make_doc("d2", "BAD contaminated " * 5),                          # contaminated
            _make_doc("d3", "good clean document number three with words " * 3),
            _make_doc("d4", "good clean document number three with words " * 3),  # dup of d3
            _make_doc("d5", "completely unique fifth document content here " * 3),
        ]
        def keep_long(doc): return len(doc.text.split()) >= 3

        stats, _ = build_one_source(
            source_id="synth",
            docs=docs,
            tokenizer=_StubTokenizer(),
            output_dir=tmp_path / "c",
            sequence_length=32,
            sequences_per_shard=2,
            revision_id="r",
            tokenizer_sha256="x",
            eos_token_id=1,
            filter_fn=keep_long,
            decontaminator=_StubDecon(),
            dedupe_config=FINEWEB_MINHASH_CONFIG,
            drop_last=False,
        )
        assert stats.docs_seen == 5
        # Each stage exits early on first failure, so docs_kept + each
        # rejection counter add up to docs_seen.
        assert (
            stats.docs_kept
            + stats.docs_filtered
            + stats.docs_contaminated
            + stats.docs_deduped
        ) == stats.docs_seen


# --------------------------------------------------------------------------- #
# Cross-source isolation — dedupe must NOT cross sources
# --------------------------------------------------------------------------- #
class TestPerSourceDedupeIsolation:
    """FineWeb finding: dedupe is per-source, not global. Each invocation
    of build_one_source builds its own MinHash index — identical docs in
    different sources are NOT deduplicated against each other."""

    def test_identical_doc_in_different_sources_kept_in_both(self, tmp_path):
        same_text = "this exact text appears in both sources " * 10
        docs_a = [_make_doc("a1", same_text)]
        docs_b = [_make_doc("b1", same_text)]

        stats_a, _ = build_one_source(
            source_id="src_a",
            docs=docs_a,
            tokenizer=_StubTokenizer(),
            output_dir=tmp_path / "src_a",
            sequence_length=32,
            sequences_per_shard=2,
            revision_id="r",
            tokenizer_sha256="x",
            eos_token_id=1,
            dedupe_config=FINEWEB_MINHASH_CONFIG,
            drop_last=False,
        )
        stats_b, _ = build_one_source(
            source_id="src_b",
            docs=docs_b,
            tokenizer=_StubTokenizer(),
            output_dir=tmp_path / "src_b",
            sequence_length=32,
            sequences_per_shard=2,
            revision_id="r",
            tokenizer_sha256="x",
            eos_token_id=1,
            dedupe_config=FINEWEB_MINHASH_CONFIG,
            drop_last=False,
        )
        # Both sources kept the doc — per-source isolation works.
        assert stats_a.docs_kept == 1
        assert stats_b.docs_kept == 1


# --------------------------------------------------------------------------- #
# Hash helpers
# --------------------------------------------------------------------------- #
class TestTextHash:
    def test_text_hash_is_deterministic(self):
        assert _text_hash("hello world") == _text_hash("hello world")

    def test_text_hash_fits_int64(self):
        for s in ["", "a", "hello world", "x" * 10_000]:
            h = _text_hash(s)
            assert 0 <= h < (1 << 63), f"hash overflowed int64 range: {h}"


# --------------------------------------------------------------------------- #
# FineWeb MinHash config sanity (locks the deployed values)
# --------------------------------------------------------------------------- #
class TestFineWebMinHashConfig:
    def test_finewebconfig_matches_published_values(self):
        """112 perms × 14 bands × 8 rows/band ≈ 0.75 Jaccard threshold
        per FineWeb / DataTrove deployed settings."""
        c = FINEWEB_MINHASH_CONFIG
        assert c.num_perm == 112
        assert c.num_bands == 14
        assert c.rows_per_band == 8
        assert c.ngram_size == 5
        assert c.threshold == 0.75


# --------------------------------------------------------------------------- #
# Batched tokenization — the perf-critical refactor
# --------------------------------------------------------------------------- #
class _BatchAwareStubTokenizer:
    """Tokenizer stub that records whether encode_batch was actually used.

    Lets us assert the build pipeline calls encode_batch (Rust Rayon path)
    and not encode (single-core path).
    """

    def __init__(self):
        self.encode_calls = 0
        self.encode_batch_calls = 0
        self.encode_batch_max_size = 0

    def encode(self, text: str):
        self.encode_calls += 1
        return _StubEncoding([hash(text) & 0xFFFF])

    def encode_batch(self, texts: list[str]):
        self.encode_batch_calls += 1
        self.encode_batch_max_size = max(self.encode_batch_max_size, len(texts))
        return [_StubEncoding([hash(t) & 0xFFFF]) for t in texts]


class TestBatchedTokenization:
    def test_uses_encode_batch_when_available(self, tmp_path):
        """The refactored pipeline must call encode_batch for the hot path,
        NOT encode-per-doc. Single-call encode is the slow path that wastes
        all cores beyond the first."""
        tok = _BatchAwareStubTokenizer()
        docs = [_make_doc(f"d{i}", f"document {i} text more text " * 5)
                for i in range(50)]
        build_one_source(
            source_id="synth",
            docs=docs,
            tokenizer=tok,
            output_dir=tmp_path / "c",
            sequence_length=32,
            sequences_per_shard=2,
            revision_id="r",
            tokenizer_sha256="x",
            eos_token_id=1,
            dedupe_config=None,
            drop_last=False,
            tokenize_batch_size=10,  # 50 docs / 10 = 5 batches
        )
        # 50 docs ÷ batch=10 → 5 calls to encode_batch
        # encode() should NOT have been called on the hot path.
        assert tok.encode_batch_calls == 5
        assert tok.encode_calls == 0
        assert tok.encode_batch_max_size == 10

    def test_partial_final_batch_flushed(self, tmp_path):
        """If the doc stream ends mid-batch, the partial must still flush."""
        tok = _BatchAwareStubTokenizer()
        docs = [_make_doc(f"d{i}", f"text words words {i} " * 5) for i in range(7)]
        stats, _ = build_one_source(
            source_id="synth", docs=docs, tokenizer=tok,
            output_dir=tmp_path / "c", sequence_length=32, sequences_per_shard=2,
            revision_id="r", tokenizer_sha256="x", eos_token_id=1,
            dedupe_config=None, drop_last=False,
            tokenize_batch_size=3,  # 7 docs → batch sizes 3, 3, 1
        )
        # 3 batches: two of size 3, one of size 1.
        assert tok.encode_batch_calls == 3
        assert stats.docs_kept == 7

    def test_falls_back_to_encode_when_encode_batch_missing(self, tmp_path):
        """Stub tokenizers (in some tests) only have .encode. The pipeline
        must still work — slowly — without crashing."""
        # The original _StubTokenizer above has only .encode (no .encode_batch).
        tok = _StubTokenizer()
        docs = [_make_doc(f"d{i}", f"document {i} text more text " * 5)
                for i in range(5)]
        stats, _ = build_one_source(
            source_id="synth", docs=docs, tokenizer=tok,
            output_dir=tmp_path / "c", sequence_length=32, sequences_per_shard=2,
            revision_id="r", tokenizer_sha256="x", eos_token_id=1,
            dedupe_config=None, drop_last=False,
        )
        assert stats.docs_kept == 5

    def test_batch_size_does_not_change_output_tokens(self, tmp_path):
        """Same input + same tokenizer, different batch sizes → identical
        output tokens. The refactor must not change semantics."""
        tok = _BatchAwareStubTokenizer()
        docs = [_make_doc(f"d{i}", f"doc {i} content with words " * 5)
                for i in range(20)]

        # Build with batch_size=5 then with batch_size=20.
        build_one_source(
            source_id="a", docs=list(docs), tokenizer=tok,
            output_dir=tmp_path / "out_5", sequence_length=32, sequences_per_shard=4,
            revision_id="r", tokenizer_sha256="x", eos_token_id=1,
            dedupe_config=None, drop_last=False, tokenize_batch_size=5,
        )
        build_one_source(
            source_id="a", docs=list(docs), tokenizer=tok,
            output_dir=tmp_path / "out_20", sequence_length=32, sequences_per_shard=4,
            revision_id="r", tokenizer_sha256="x", eos_token_id=1,
            dedupe_config=None, drop_last=False, tokenize_batch_size=20,
        )
        # Compare the produced corpora's tokens byte-exact.
        from myllm.data.packed_corpus import PackedCorpusReader

        r5 = PackedCorpusReader(tmp_path / "out_5")
        r20 = PackedCorpusReader(tmp_path / "out_20")
        assert r5.total_sequences == r20.total_sequences
        for sid in range(r5.total_sequences):
            np.testing.assert_array_equal(
                r5.get_sequence(sid), r20.get_sequence(sid),
            )
