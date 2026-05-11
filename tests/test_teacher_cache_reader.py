"""Tests for the runtime TeacherCacheReader (R0 follow-up, 2026-05-11 audit).

Validates:
  - Random-access lookups: ``get_topk(positions)`` round-trips correctly
    for positions spread across multiple shards.
  - Coverage queries: ``coverage_range()`` and ``has_coverage()``.
  - Out-of-coverage positions raise a clear error.
  - LRU shard cache evicts old shards as new ones are touched.
  - SHA-256 verification (opt-in).
  - MultiTeacherCacheReader fans across teachers and stacks the output.

Uses the existing ``write_shard`` / ``write_manifest`` helpers to populate
a small fake cache in a tmp dir. No real teacher needed.
"""
from __future__ import annotations

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pa = pytest.importorskip("pyarrow")

from myllm.data.teacher_cache import (
    CacheManifest,
    CacheShard,
    MultiTeacherCacheReader,
    ShardManifestEntry,
    TeacherCacheReader,
    compute_shard_key,
    write_manifest,
    write_shard,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _f32_to_bf16_bits(f32: np.ndarray) -> np.ndarray:
    """Truncate float32 to bf16 by keeping the high 16 bits. Mirrors the
    producer's encoding in ``scripts/cache_teacher_logits.py``."""
    return (f32.view("uint32").astype("uint32") >> 16).astype("uint16")


def _bf16_bits_to_f32(u16: np.ndarray) -> np.ndarray:
    """Inverse of ``_f32_to_bf16_bits``. Mirrors the reader's decoder.
    See ``myllm.data.teacher_cache._bf16_bits_to_f32``."""
    u32 = u16.astype(np.uint32) << np.uint32(16)
    return u32.view(np.float32)


def _write_synthetic_cache(
    tmp_path: Path,
    teacher_id: str,
    n_shards: int = 3,
    tokens_per_shard: int = 5,
    top_k: int = 4,
    corpus_sha: str = "a" * 64,
    tokenizer_sha: str = "b" * 64,
    base_seed: int = 0,
) -> tuple[Path, dict[int, tuple[np.ndarray, np.ndarray]]]:
    """Write ``n_shards`` consecutive shards. Return (manifest_path, position_to_value_map).

    ``position_to_value_map[pos] = (logits[K] float32, indices[K] uint32)``
    is the ground truth for round-trip comparisons. Logits are stored as
    bf16 bit patterns on disk (matching the producer) but the truth dict
    stores them as float32 (matching what the reader returns post-bitcast).
    Tests verify the f32 → bf16 → f32 round-trip preserves bit-equivalent
    values to within bf16 precision (truncation, ~0.4% relative error).
    """
    output_dir = tmp_path / "cache"
    manifest = CacheManifest(
        teacher_id=teacher_id,
        corpus_sha256=corpus_sha,
        tokenizer_sha256=tokenizer_sha,
        top_k=top_k,
    )
    truth: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    rng = np.random.default_rng(seed=base_seed)
    for i in range(n_shards):
        start = i * tokens_per_shard
        end = start + tokens_per_shard
        # Generate realistic-range f32 logits (LLM outputs are typically in
        # [-10, 10] range pre-softmax) then truncate to bf16 bits.
        logits_f32 = rng.normal(0.0, 4.0, size=(tokens_per_shard, top_k)).astype("float32")
        logits_u16 = _f32_to_bf16_bits(logits_f32)
        # Truth = the bf16-rounded f32 values the reader will return.
        logits_truth = _bf16_bits_to_f32(logits_u16)

        indices = rng.integers(0, 131072, size=(tokens_per_shard, top_k), dtype="uint32")
        for j in range(tokens_per_shard):
            truth[start + j] = (logits_truth[j].copy(), indices[j].copy())
        shard = CacheShard(
            teacher_id=teacher_id,
            corpus_sha256=corpus_sha,
            tokenizer_sha256=tokenizer_sha,
            start_token_position=start,
            end_token_position=end,
            top_k=top_k,
            logits=logits_u16,
            indices=indices,
        )
        key = compute_shard_key(teacher_id, top_k, corpus_sha, start, end)
        local = output_dir / key
        sha = write_shard(shard, local)
        manifest.shards.append(ShardManifestEntry(start, end, key, sha))
    manifest_path = output_dir / f"{teacher_id}_manifest.json"
    write_manifest(manifest, manifest_path)
    return manifest_path, truth


# --------------------------------------------------------------------------- #
# Coverage queries (no I/O)
# --------------------------------------------------------------------------- #
class TestCoverageQueries:
    def test_coverage_range_spans_all_shards(self, tmp_path):
        mp, _ = _write_synthetic_cache(tmp_path, "t", n_shards=3, tokens_per_shard=10)
        reader = TeacherCacheReader("t", mp, tmp_path / "cache")
        assert reader.coverage_range() == (0, 30)

    def test_coverage_range_empty_manifest(self, tmp_path):
        manifest = CacheManifest(teacher_id="t", corpus_sha256="0"*64, tokenizer_sha256="0"*64, top_k=4)
        manifest_path = tmp_path / "empty_t_manifest.json"
        write_manifest(manifest, manifest_path)
        reader = TeacherCacheReader("t", manifest_path, tmp_path)
        assert reader.coverage_range() == (0, 0)

    def test_has_coverage_subrange(self, tmp_path):
        mp, _ = _write_synthetic_cache(tmp_path, "t", n_shards=3, tokens_per_shard=10)
        reader = TeacherCacheReader("t", mp, tmp_path / "cache")
        assert reader.has_coverage(0, 30) is True
        assert reader.has_coverage(5, 25) is True
        assert reader.has_coverage(0, 31) is False  # beyond end
        assert reader.has_coverage(10, 100) is False


# --------------------------------------------------------------------------- #
# Random-access reads
# --------------------------------------------------------------------------- #
class TestGetTopK:
    def test_single_position_round_trip(self, tmp_path):
        mp, truth = _write_synthetic_cache(tmp_path, "t", n_shards=2, tokens_per_shard=10, top_k=4)
        reader = TeacherCacheReader("t", mp, tmp_path / "cache")

        logits, indices = reader.get_topk(np.array([7], dtype=np.int64))
        assert logits.shape == (1, 4)
        assert indices.shape == (1, 4)
        np.testing.assert_array_equal(logits[0], truth[7][0])
        np.testing.assert_array_equal(indices[0], truth[7][1])

    def test_batched_positions_within_one_shard(self, tmp_path):
        mp, truth = _write_synthetic_cache(tmp_path, "t", n_shards=2, tokens_per_shard=10, top_k=3)
        reader = TeacherCacheReader("t", mp, tmp_path / "cache")

        positions = np.array([3, 7, 1, 9], dtype=np.int64)
        logits, indices = reader.get_topk(positions)
        assert logits.shape == (4, 3)
        for i, pos in enumerate(positions):
            np.testing.assert_array_equal(logits[i], truth[int(pos)][0])
            np.testing.assert_array_equal(indices[i], truth[int(pos)][1])

    def test_batched_positions_across_multiple_shards(self, tmp_path):
        mp, truth = _write_synthetic_cache(tmp_path, "t", n_shards=4, tokens_per_shard=8, top_k=2)
        reader = TeacherCacheReader("t", mp, tmp_path / "cache")

        # Positions spread across all 4 shards, in mixed order.
        positions = np.array([0, 15, 25, 7, 31, 8, 16, 23], dtype=np.int64)
        logits, indices = reader.get_topk(positions)
        for i, pos in enumerate(positions):
            np.testing.assert_array_equal(logits[i], truth[int(pos)][0])
            np.testing.assert_array_equal(indices[i], truth[int(pos)][1])

    def test_empty_positions_returns_empty_arrays(self, tmp_path):
        mp, _ = _write_synthetic_cache(tmp_path, "t", n_shards=1, tokens_per_shard=5)
        reader = TeacherCacheReader("t", mp, tmp_path / "cache")

        logits, indices = reader.get_topk(np.array([], dtype=np.int64))
        assert logits.shape == (0, reader.top_k)
        assert indices.shape == (0, reader.top_k)

    def test_logits_returned_as_float32_not_uint16(self, tmp_path):
        """Regression for P0-5 from 2026-05-12 audit. The cache stores
        teacher logits as bf16 bit patterns packed into uint16. If the
        reader returns those uint16 ints raw, every downstream consumer
        (KL loss, softmax) would compute math on integers 0-65535 instead
        of real float values — silently corrupting distillation.

        get_topk MUST return float32, not uint16. The bf16 round-trip
        through the truth dict should preserve values to bf16 precision."""
        mp, truth = _write_synthetic_cache(tmp_path, "t", n_shards=2,
                                            tokens_per_shard=10, top_k=4)
        reader = TeacherCacheReader("t", mp, tmp_path / "cache")

        logits, indices = reader.get_topk(np.array([0, 5, 12, 18], dtype=np.int64))
        # Dtype check is the actual regression assertion — uint16 = fail.
        assert logits.dtype == np.float32, (
            f"reader must return float32 logits, got {logits.dtype}. "
            f"If this fails, the bf16-bits-to-f32 conversion at the reader "
            f"boundary is missing — see _bf16_bits_to_f32 in teacher_cache.py."
        )
        assert indices.dtype == np.uint32
        # Values should be in a sensible range (not 0-65535 integer territory)
        # for normal-distributed logits. uint16 values would be huge ints.
        assert np.all(np.abs(logits) < 100), (
            f"logits look like uint16 integers, not float32 — got range "
            f"[{logits.min()}, {logits.max()}]"
        )

    def test_bf16_round_trip_preserves_values(self, tmp_path):
        """bf16 has ~3 decimal digits of precision (7-bit mantissa + sign + 8-bit
        exponent). f32 → bf16 → f32 should be within ~0.4% relative error
        for values away from zero."""
        mp, truth = _write_synthetic_cache(tmp_path, "t", n_shards=1,
                                            tokens_per_shard=8, top_k=3,
                                            base_seed=42)
        reader = TeacherCacheReader("t", mp, tmp_path / "cache")
        positions = np.arange(8, dtype=np.int64)
        logits, _ = reader.get_topk(positions)

        for i in range(8):
            np.testing.assert_allclose(
                logits[i], truth[int(positions[i])][0],
                rtol=0.0, atol=0.0,  # exact equality — truth ALSO went through bf16 round-trip
                err_msg=f"position {i}: reader f32 must match the truth f32 "
                        f"(both are post-bf16-truncation)",
            )

    def test_out_of_range_position_raises(self, tmp_path):
        mp, _ = _write_synthetic_cache(tmp_path, "t", n_shards=2, tokens_per_shard=5)
        reader = TeacherCacheReader("t", mp, tmp_path / "cache")

        # Cache covers [0, 10). Position 100 is out of range.
        with pytest.raises(ValueError, match="outside cache coverage"):
            reader.get_topk(np.array([100], dtype=np.int64))

    def test_negative_position_raises(self, tmp_path):
        mp, _ = _write_synthetic_cache(tmp_path, "t", n_shards=2, tokens_per_shard=5)
        reader = TeacherCacheReader("t", mp, tmp_path / "cache")

        with pytest.raises(ValueError, match="outside cache coverage"):
            reader.get_topk(np.array([-1], dtype=np.int64))

    def test_2d_positions_rejected(self, tmp_path):
        mp, _ = _write_synthetic_cache(tmp_path, "t", n_shards=1, tokens_per_shard=5)
        reader = TeacherCacheReader("t", mp, tmp_path / "cache")
        with pytest.raises(ValueError, match="1-D"):
            reader.get_topk(np.array([[0, 1], [2, 3]], dtype=np.int64))


# --------------------------------------------------------------------------- #
# Manifest mismatch
# --------------------------------------------------------------------------- #
class TestManifestValidation:
    def test_wrong_teacher_id_raises(self, tmp_path):
        mp, _ = _write_synthetic_cache(tmp_path, "teacher-A", n_shards=1, tokens_per_shard=5)
        with pytest.raises(ValueError, match="not 'teacher-B'"):
            TeacherCacheReader("teacher-B", mp, tmp_path / "cache")

    def test_missing_shard_file_raises_on_access(self, tmp_path):
        mp, _ = _write_synthetic_cache(tmp_path, "t", n_shards=1, tokens_per_shard=5)
        reader = TeacherCacheReader("t", mp, tmp_path / "cache")
        # Delete the underlying shard file.
        for f in (tmp_path / "cache").rglob("*.arrow"):
            f.unlink()
        with pytest.raises(FileNotFoundError, match="not present locally"):
            reader.get_topk(np.array([0], dtype=np.int64))


# --------------------------------------------------------------------------- #
# Sha-256 verification
# --------------------------------------------------------------------------- #
class TestShaVerification:
    def test_verify_sha256_passes_for_intact_shard(self, tmp_path):
        mp, _ = _write_synthetic_cache(tmp_path, "t", n_shards=1, tokens_per_shard=5)
        reader = TeacherCacheReader("t", mp, tmp_path / "cache", verify_sha256=True)
        # Should not raise.
        reader.get_topk(np.array([0], dtype=np.int64))


# --------------------------------------------------------------------------- #
# LRU eviction
# --------------------------------------------------------------------------- #
class TestLRUEviction:
    def test_open_shards_bounded_by_max_open_shards(self, tmp_path):
        mp, _ = _write_synthetic_cache(tmp_path, "t", n_shards=5, tokens_per_shard=4)
        reader = TeacherCacheReader("t", mp, tmp_path / "cache", max_open_shards=2)

        # Touch all 5 shards; only the most recent 2 should remain open.
        for pos in [0, 4, 8, 12, 16]:  # 5 different shards
            reader.get_topk(np.array([pos], dtype=np.int64))
        assert len(reader._open) <= 2  # noqa: SLF001 — internal state inspection in test


# --------------------------------------------------------------------------- #
# Multi-teacher fan-out
# --------------------------------------------------------------------------- #
class TestMultiTeacher:
    def test_stack_across_teachers(self, tmp_path):
        mp_a, _ = _write_synthetic_cache(tmp_path / "a", "ta", n_shards=2, tokens_per_shard=4, top_k=3, base_seed=1)
        mp_b, _ = _write_synthetic_cache(tmp_path / "b", "tb", n_shards=2, tokens_per_shard=4, top_k=3, base_seed=2)
        r_a = TeacherCacheReader("ta", mp_a, tmp_path / "a" / "cache")
        r_b = TeacherCacheReader("tb", mp_b, tmp_path / "b" / "cache")
        multi = MultiTeacherCacheReader([r_a, r_b])

        positions = np.array([0, 3, 5], dtype=np.int64)
        logits, indices = multi.get_topk(positions)
        # Leading axis is teacher count.
        assert logits.shape == (2, 3, 3)
        assert indices.shape == (2, 3, 3)
        # Per-teacher slices match each reader's individual output.
        for t, r in enumerate([r_a, r_b]):
            lg, ix = r.get_topk(positions)
            np.testing.assert_array_equal(logits[t], lg)
            np.testing.assert_array_equal(indices[t], ix)

    def test_top_k_mismatch_rejected(self, tmp_path):
        mp_a, _ = _write_synthetic_cache(tmp_path / "a", "ta", n_shards=1, tokens_per_shard=4, top_k=4)
        mp_b, _ = _write_synthetic_cache(tmp_path / "b", "tb", n_shards=1, tokens_per_shard=4, top_k=8)
        r_a = TeacherCacheReader("ta", mp_a, tmp_path / "a" / "cache")
        r_b = TeacherCacheReader("tb", mp_b, tmp_path / "b" / "cache")
        with pytest.raises(ValueError, match="top_k"):
            MultiTeacherCacheReader([r_a, r_b])

    def test_empty_reader_list_rejected(self):
        with pytest.raises(ValueError, match=">= 1"):
            MultiTeacherCacheReader([])

    def test_teacher_ids_property(self, tmp_path):
        mp_a, _ = _write_synthetic_cache(tmp_path / "a", "ta", n_shards=1, tokens_per_shard=4, top_k=4)
        mp_b, _ = _write_synthetic_cache(tmp_path / "b", "tb", n_shards=1, tokens_per_shard=4, top_k=4)
        r_a = TeacherCacheReader("ta", mp_a, tmp_path / "a" / "cache")
        r_b = TeacherCacheReader("tb", mp_b, tmp_path / "b" / "cache")
        multi = MultiTeacherCacheReader([r_a, r_b])
        assert multi.teacher_ids == ["ta", "tb"]
