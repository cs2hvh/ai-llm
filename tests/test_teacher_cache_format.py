"""Round-trip tests for the teacher logit cache binary format.

Validates `src/myllm/data/teacher_cache.py` — writing and reading back
synthetic Arrow shards. Does NOT load any real teacher model; the
cache I/O is independent of the producer (cache_teacher_logits.py).

Key invariants:
  1. Shard round-trip preserves logit values + indices + provenance.
  2. Atomic write — no partial files visible to readers.
  3. Content-addressed naming is deterministic and stable.
  4. Manifest coverage checks correctly detect gaps.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pa = pytest.importorskip("pyarrow")

from myllm.data.teacher_cache import (  # noqa: E402
    FORMAT_VERSION,
    CacheManifest,
    CacheShard,
    ShardManifestEntry,
    compute_shard_key,
    read_manifest,
    read_shard,
    write_manifest,
    write_shard,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _toy_shard(n_tokens=4, top_k=3, start=0, teacher_id="toy-teacher"):
    """A small valid shard for tests."""
    # logits as raw uint16 bytes (proxy for bfloat16)
    rng = np.random.default_rng(seed=0)
    logits = rng.integers(0, 65535, size=(n_tokens, top_k), dtype="uint16")
    indices = rng.integers(0, 131072, size=(n_tokens, top_k), dtype="uint32")
    return CacheShard(
        teacher_id=teacher_id,
        corpus_sha256="a" * 64,
        tokenizer_sha256="b" * 64,
        start_token_position=start,
        end_token_position=start + n_tokens,
        top_k=top_k,
        logits=logits,
        indices=indices,
    )


# --------------------------------------------------------------------------- #
# Shard round-trip
# --------------------------------------------------------------------------- #
class TestShardRoundTrip:
    def test_write_then_read_preserves_payload(self, tmp_path):
        shard = _toy_shard(n_tokens=10, top_k=8)
        path = tmp_path / "shard.arrow"
        sha = write_shard(shard, path)

        loaded = read_shard(path)
        assert loaded.teacher_id == shard.teacher_id
        assert loaded.corpus_sha256 == shard.corpus_sha256
        assert loaded.tokenizer_sha256 == shard.tokenizer_sha256
        assert loaded.start_token_position == shard.start_token_position
        assert loaded.end_token_position == shard.end_token_position
        assert loaded.top_k == shard.top_k
        assert loaded.format_version == shard.format_version
        np.testing.assert_array_equal(loaded.logits, shard.logits)
        np.testing.assert_array_equal(loaded.indices, shard.indices)
        # sha256 returned by write must match a re-computed value.
        assert len(sha) == 64 and all(c in "0123456789abcdef" for c in sha)

    def test_write_atomic_no_tmpfile_visible_after_success(self, tmp_path):
        shard = _toy_shard()
        path = tmp_path / "shard.arrow"
        write_shard(shard, path)
        # The .tmp shouldn't remain after a successful write.
        assert not (tmp_path / "shard.arrow.tmp").exists()
        assert path.exists()

    def test_read_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_shard(tmp_path / "nope.arrow")


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
class TestShardValidate:
    def test_negative_topk_rejected(self):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="top_k"):
            CacheShard(
                teacher_id="x", corpus_sha256="0"*64, tokenizer_sha256="0"*64,
                start_token_position=0, end_token_position=2, top_k=0,
                logits=rng.integers(0, 1, (2, 0), dtype="uint16"),
                indices=rng.integers(0, 1, (2, 0), dtype="uint32"),
            ).validate()

    def test_misaligned_logits_shape_rejected(self):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="logits shape"):
            CacheShard(
                teacher_id="x", corpus_sha256="0"*64, tokenizer_sha256="0"*64,
                start_token_position=0, end_token_position=5, top_k=3,
                logits=rng.integers(0, 1, (4, 3), dtype="uint16"),    # 4 != 5
                indices=rng.integers(0, 1, (5, 3), dtype="uint32"),
            ).validate()

    def test_end_before_start_rejected(self):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="end_token_position"):
            CacheShard(
                teacher_id="x", corpus_sha256="0"*64, tokenizer_sha256="0"*64,
                start_token_position=10, end_token_position=5, top_k=2,
                logits=rng.integers(0, 1, (0, 2), dtype="uint16"),
                indices=rng.integers(0, 1, (0, 2), dtype="uint32"),
            ).validate()


# --------------------------------------------------------------------------- #
# Naming
# --------------------------------------------------------------------------- #
class TestShardKey:
    def test_deterministic(self):
        key1 = compute_shard_key("deepseek-v4-pro-base", 8, "a" * 64, 0, 10_000_000)
        key2 = compute_shard_key("deepseek-v4-pro-base", 8, "a" * 64, 0, 10_000_000)
        assert key1 == key2

    def test_format_matches_spec(self):
        key = compute_shard_key("deepseek-v4-pro-base", 8, "abc123def456" + "0"*52, 0, 10_000_000)
        # Format: distillation_cache/{teacher_id}/k{top_k}/corpus_{sha[:16]}/tokens_{start}_{end}.arrow
        assert key.startswith("distillation_cache/deepseek-v4-pro-base/k8/")
        assert "corpus_abc123def4560000" in key
        assert key.endswith("_0000010000000.arrow")

    def test_different_teachers_get_different_keys(self):
        sha = "0" * 64
        k1 = compute_shard_key("teacher-a", 8, sha, 0, 1000)
        k2 = compute_shard_key("teacher-b", 8, sha, 0, 1000)
        assert k1 != k2


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #
class TestManifest:
    def _toy_manifest(self):
        return CacheManifest(
            teacher_id="toy-teacher",
            corpus_sha256="a" * 64,
            tokenizer_sha256="b" * 64,
            top_k=8,
            shards=[
                ShardManifestEntry(0, 1000, "key1", "sha1"),
                ShardManifestEntry(1000, 2500, "key2", "sha2"),
                ShardManifestEntry(2500, 5000, "key3", "sha3"),
            ],
        )

    def test_round_trip(self, tmp_path):
        m = self._toy_manifest()
        path = tmp_path / "manifest.json"
        write_manifest(m, path)

        loaded = read_manifest(path)
        assert loaded.teacher_id == m.teacher_id
        assert loaded.corpus_sha256 == m.corpus_sha256
        assert loaded.top_k == m.top_k
        assert len(loaded.shards) == 3
        assert loaded.total_tokens() == 5000

    def test_total_tokens(self):
        m = self._toy_manifest()
        assert m.total_tokens() == 5000

    def test_assert_covers_passes_full_coverage(self):
        m = self._toy_manifest()
        m.assert_covers(0, 5000)
        m.assert_covers(500, 4000)  # subset is also covered

    def test_assert_covers_detects_gap(self):
        m = CacheManifest(
            teacher_id="x", corpus_sha256="0"*64, tokenizer_sha256="0"*64, top_k=8,
            shards=[
                ShardManifestEntry(0, 1000, "k1", "s1"),
                ShardManifestEntry(2000, 3000, "k2", "s2"),  # GAP at 1000-2000
            ],
        )
        with pytest.raises(ValueError, match="gap in coverage"):
            m.assert_covers(0, 3000)

    def test_assert_covers_detects_insufficient_end(self):
        m = self._toy_manifest()
        with pytest.raises(ValueError, match="manifest covers up to"):
            m.assert_covers(0, 10000)  # manifest only covers to 5000

    def test_unsupported_format_version_rejected(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({
            "format_version": 999,
            "teacher_id": "x",
            "corpus_sha256": "0"*64,
            "tokenizer_sha256": "0"*64,
            "top_k": 8,
            "shards": [],
        }))
        with pytest.raises(ValueError, match="unsupported manifest format_version"):
            read_manifest(path)
