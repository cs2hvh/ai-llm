"""Teacher logit cache — Arrow shard I/O.

Binary format spec lives in ``docs/teacher_cache.py``. This
module is the implementation:

  - ``CacheShard``           — typed wrapper around one shard's payload.
  - ``write_shard``          — atomic write of an Arrow IPC stream.
  - ``read_shard``           — load a shard from disk (or via mmap).
  - ``compute_shard_key``    — deterministic R2 key from
                               ``(teacher_id, top_k, corpus_sha, start, end)``.
  - ``CacheManifest``        — per-teacher manifest: list of shards + provenance.
  - ``write_manifest`` / ``read_manifest`` — atomic JSON I/O.

R0 (2026-05-11). Producer (``scripts/cache_teacher_logits.py``) and
runtime reader (extends this module in a follow-up PR) both build on
top of these primitives.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# All format constants come from teacher_cache.py spec.
FORMAT_VERSION = 1
LOGIT_DTYPE = "bfloat16"   # 2 bytes per logit value
INDEX_DTYPE = "uint32"     # 4 bytes per vocab index


# --------------------------------------------------------------------------- #
# Shard payload
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CacheShard:
    """One shard of teacher top-K logits, covering a contiguous token range.

    Shape conventions:
        logits:  [n_tokens, top_k]   bfloat16
        indices: [n_tokens, top_k]   uint32

    Where ``n_tokens == end_token_position - start_token_position``.
    """

    teacher_id: str
    corpus_sha256: str            # hex string, 64 chars
    tokenizer_sha256: str         # hex string, 64 chars
    start_token_position: int
    end_token_position: int
    top_k: int
    logits: Any                   # np.ndarray [n_tokens, top_k] of bfloat16
    indices: Any                  # np.ndarray [n_tokens, top_k] of uint32
    format_version: int = FORMAT_VERSION

    def n_tokens(self) -> int:
        return self.end_token_position - self.start_token_position

    def validate(self) -> None:
        """Raise ValueError if the payload is internally inconsistent."""
        import numpy as np
        if self.top_k <= 0:
            raise ValueError(f"top_k must be > 0, got {self.top_k}")
        if self.end_token_position <= self.start_token_position:
            raise ValueError(
                f"end_token_position ({self.end_token_position}) must be > "
                f"start_token_position ({self.start_token_position})"
            )
        expected_n = self.n_tokens()
        if self.logits.shape != (expected_n, self.top_k):
            raise ValueError(
                f"logits shape {self.logits.shape} != expected ({expected_n}, {self.top_k})"
            )
        if self.indices.shape != (expected_n, self.top_k):
            raise ValueError(
                f"indices shape {self.indices.shape} != expected ({expected_n}, {self.top_k})"
            )
        if not np.issubdtype(self.indices.dtype, np.unsignedinteger):
            raise ValueError(f"indices dtype must be unsigned int, got {self.indices.dtype}")


# --------------------------------------------------------------------------- #
# Shard naming
# --------------------------------------------------------------------------- #
def compute_shard_key(
    teacher_id: str,
    top_k: int,
    corpus_sha256: str,
    start_token_position: int,
    end_token_position: int,
) -> str:
    """Deterministic R2 key under which a shard is stored.

    See ``docs/teacher_cache.py`` §"Naming".
    """
    corpus_short = corpus_sha256[:16]
    return (
        f"distillation_cache/{teacher_id}/k{top_k}/corpus_{corpus_short}/"
        f"tokens_{start_token_position:013d}_{end_token_position:013d}.arrow"
    )


def compute_local_path(root: Path, key: str) -> Path:
    """Local cache path that mirrors the R2 key structure."""
    return Path(root) / key


# --------------------------------------------------------------------------- #
# Atomic shard I/O
# --------------------------------------------------------------------------- #
def _build_arrow_table(shard: CacheShard):
    """Build a single-RecordBatch Arrow Table from a shard."""
    import pyarrow as pa

    # logits stored as bfloat16 list per row; pyarrow doesn't have a
    # first-class bfloat16 column type, so we store the raw bytes as a
    # FixedSizeBinaryArray of (top_k * 2) bytes per row, and the dtype
    # marker is in the shard metadata. uint32 indices are native.
    n = shard.n_tokens()
    logit_bytes_per_row = shard.top_k * 2  # bfloat16 = 2 bytes

    # Coerce logits to bytes. The producer should already have them in
    # bfloat16-on-disk layout; here we just take the raw bytes.
    logits_bytes = shard.logits.astype("uint16").tobytes()  # bfloat16 fits in uint16
    indices_bytes = shard.indices.astype("uint32").tobytes()

    logits_arr = pa.FixedSizeBinaryArray.from_buffers(
        pa.binary(logit_bytes_per_row),
        n,
        [None, pa.py_buffer(logits_bytes)],
    )
    indices_arr = pa.FixedSizeListArray.from_arrays(
        pa.array(shard.indices.flatten(), type=pa.uint32()),
        list_size=shard.top_k,
    )

    schema_metadata = {
        b"format_version": str(shard.format_version).encode(),
        b"teacher_id": shard.teacher_id.encode(),
        b"corpus_sha256": shard.corpus_sha256.encode(),
        b"tokenizer_sha256": shard.tokenizer_sha256.encode(),
        b"start_token_position": str(shard.start_token_position).encode(),
        b"end_token_position": str(shard.end_token_position).encode(),
        b"top_k": str(shard.top_k).encode(),
        b"logit_dtype": LOGIT_DTYPE.encode(),
        b"index_dtype": INDEX_DTYPE.encode(),
    }
    schema = pa.schema(
        [
            pa.field("logits", pa.binary(logit_bytes_per_row)),
            pa.field("indices", pa.list_(pa.uint32(), shard.top_k)),
        ],
        metadata=schema_metadata,
    )
    table = pa.table([logits_arr, indices_arr], schema=schema)
    return table


def write_shard(shard: CacheShard, path: str | Path) -> str:
    """Atomically write ``shard`` to ``path``. Returns the sha256 of the file.

    Atomic via tmpfile + rename: a reader never sees a partial file. The
    sha256 is computed *after* the rename to ensure it covers the file
    that actually exists on disk.
    """
    import pyarrow as pa

    shard.validate()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    table = _build_arrow_table(shard)
    with pa.OSFile(str(tmp), "wb") as sink:
        with pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)
    os.replace(tmp, target)
    return _sha256_file(target)


def read_shard(path: str | Path) -> CacheShard:
    """Load a shard from ``path``. Reads the full Arrow stream into memory.

    For mmap-style streaming reads use ``open_shard_mmap`` (next PR).
    """
    import numpy as np
    import pyarrow as pa

    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"shard not found: {target}")
    with pa.OSFile(str(target), "rb") as source:
        reader = pa.ipc.open_file(source)
        table = reader.read_all()

    md = {k.decode(): v.decode() for k, v in (table.schema.metadata or {}).items()}
    version = int(md.get("format_version", "0"))
    if version != FORMAT_VERSION:
        raise ValueError(
            f"unsupported format_version {version}; expected {FORMAT_VERSION}"
        )
    top_k = int(md["top_k"])
    start = int(md["start_token_position"])
    end = int(md["end_token_position"])
    n = end - start

    logits_col = table.column("logits").to_pylist()
    # logits_col is a list of bytes objects, each of length top_k * 2.
    raw = b"".join(logits_col)
    # We stored bfloat16 as uint16 byte-level (round-trip preserves bits);
    # reader returns uint16; consumer reinterprets when needed (JAX has
    # bfloat16 support, so we can view this in-place at load time).
    logits = np.frombuffer(raw, dtype="uint16").reshape(n, top_k)

    indices_arr = table.column("indices")
    indices = np.array(indices_arr.to_pylist(), dtype="uint32")

    return CacheShard(
        teacher_id=md["teacher_id"],
        corpus_sha256=md["corpus_sha256"],
        tokenizer_sha256=md["tokenizer_sha256"],
        start_token_position=start,
        end_token_position=end,
        top_k=top_k,
        logits=logits,
        indices=indices,
        format_version=version,
    )


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #
@dataclass
class ShardManifestEntry:
    start_token_position: int
    end_token_position: int
    r2_key: str
    sha256: str

    def to_dict(self) -> dict:
        return {
            "start_token_position": self.start_token_position,
            "end_token_position": self.end_token_position,
            "r2_key": self.r2_key,
            "sha256": self.sha256,
        }


@dataclass
class CacheManifest:
    """Per-teacher manifest. Listing of all shards + provenance."""

    teacher_id: str
    corpus_sha256: str
    tokenizer_sha256: str
    top_k: int
    shards: list[ShardManifestEntry] = field(default_factory=list)
    format_version: int = FORMAT_VERSION

    def total_tokens(self) -> int:
        return sum(s.end_token_position - s.start_token_position for s in self.shards)

    def to_dict(self) -> dict:
        return {
            "format_version": self.format_version,
            "teacher_id": self.teacher_id,
            "corpus_sha256": self.corpus_sha256,
            "tokenizer_sha256": self.tokenizer_sha256,
            "top_k": self.top_k,
            "total_tokens": self.total_tokens(),
            "shards": [s.to_dict() for s in self.shards],
        }

    def assert_covers(self, start: int, end: int) -> None:
        """Raise ValueError if any token in ``[start, end)`` is uncovered."""
        # Sort by start; walk forward checking coverage.
        sorted_shards = sorted(self.shards, key=lambda s: s.start_token_position)
        cursor = start
        for s in sorted_shards:
            if s.end_token_position <= cursor:
                continue
            if s.start_token_position > cursor:
                raise ValueError(
                    f"gap in coverage at token {cursor}: next shard starts at "
                    f"{s.start_token_position}"
                )
            cursor = s.end_token_position
            if cursor >= end:
                return
        if cursor < end:
            raise ValueError(
                f"manifest covers up to token {cursor}, but needs to cover {end}"
            )


def write_manifest(manifest: CacheManifest, path: str | Path) -> None:
    """Atomic JSON write."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest.to_dict(), indent=2))
    os.replace(tmp, target)


def read_manifest(path: str | Path) -> CacheManifest:
    target = Path(path)
    data = json.loads(target.read_text())
    if data.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"unsupported manifest format_version {data.get('format_version')}; "
            f"expected {FORMAT_VERSION}"
        )
    return CacheManifest(
        teacher_id=data["teacher_id"],
        corpus_sha256=data["corpus_sha256"],
        tokenizer_sha256=data["tokenizer_sha256"],
        top_k=data["top_k"],
        shards=[
            ShardManifestEntry(
                start_token_position=s["start_token_position"],
                end_token_position=s["end_token_position"],
                r2_key=s["r2_key"],
                sha256=s["sha256"],
            )
            for s in data["shards"]
        ],
        format_version=data["format_version"],
    )


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute sha256 of a file. Streams in chunks to avoid loading 38GB."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Runtime cache reader — `TeacherCacheReader.get_topk(positions)`
#
# R0 follow-up (2026-05-11 audit): the format spec at
# `docs/teacher_cache.py` §"Reader contract" promised a
# random-access ``get_topk`` interface. Implemented here. mmap-backed so
# 21.6 TB of cached logits across three teachers can be queried without
# loading any shard into RAM beyond the rows touched by the current batch.
# --------------------------------------------------------------------------- #
import bisect


def _bf16_bits_to_f32(u16: Any) -> Any:
    """Reinterpret a uint16 array of bfloat16 bit patterns as float32.

    bfloat16 is a truncated IEEE 754 single-precision float — the high 16
    bits of the f32 layout. To bitcast back to f32:
        u32 = (u16 << 16)    # place bf16 bits in the high half
        f32 = bitcast(u32)

    Why this exists:
        The cache stores teacher logits as raw bf16 bits packed into uint16
        (pyarrow has no first-class bf16 column type). If the consumer
        skips this conversion and divides those uint16 ints by temperature,
        the softmax math is meaningless — bf16(1.0) is the uint16 value
        16256, not 1.0. The audit caught exactly that bug. This helper
        makes the reader's return type unambiguous: real float32 always.
    """
    import numpy as np

    u16 = np.asarray(u16, dtype=np.uint16)
    # Shift the bf16 bits into the high 16 bits of a uint32, then bitcast
    # to float32. Equivalent to ml_dtypes.bfloat16 view + .astype(float32),
    # but avoids the optional ml_dtypes dependency.
    u32 = u16.astype(np.uint32) << np.uint32(16)
    return u32.view(np.float32)


class TeacherCacheReader:
    """Random-access reader over a single teacher's logit cache.

    One reader per teacher. To read from multiple teachers, instantiate one
    reader each (or use :class:`MultiTeacherCacheReader` below).

    Lookups:
        Given a 1-D array of corpus token positions ``pos[N]``, returns a
        pair of arrays ``(logits[N, K] float32, indices[N, K] uint32)``.
        The on-disk format stores logits as bf16 bit patterns packed into
        uint16; ``get_topk`` reinterprets them as float32 before return.
        Consumers should never see the uint16 representation.

    Memory model:
        - Shards are mmap'd on first access via ``pyarrow.memory_map`` — no
          copy into Python heap.
        - An LRU cache (default 8 shards per reader) keeps recently-touched
          shards open. Older shards are closed automatically as new ones
          page in.
        - For training: 6M+ positions per batch is fine; positions tend to
          be sequential so adjacent batches hit the same shard.
    """

    def __init__(
        self,
        teacher_id: str,
        manifest_path: str | Path,
        shard_root: str | Path,
        *,
        max_open_shards: int = 8,
        verify_sha256: bool = False,
    ):
        self.teacher_id = teacher_id
        self.shard_root = Path(shard_root)
        self.max_open_shards = max_open_shards
        self.verify_sha256 = verify_sha256

        self._manifest = read_manifest(manifest_path)
        if self._manifest.teacher_id != teacher_id:
            raise ValueError(
                f"manifest is for teacher {self._manifest.teacher_id!r}, "
                f"not {teacher_id!r}"
            )
        # Sorted list of shard entries, indexed by start_token_position.
        self._entries: list[ShardManifestEntry] = sorted(
            self._manifest.shards, key=lambda s: s.start_token_position
        )
        self._starts: list[int] = [s.start_token_position for s in self._entries]
        # LRU of opened shards: maps r2_key → (mmap, table). Insertion order
        # of dict = MRU order; we pop from the front when over max.
        self._open: dict[str, tuple[Any, Any]] = {}
        self.top_k = self._manifest.top_k

    # ------------------------------------------------------------------- #
    # Coverage queries (cheap — no shard I/O needed)
    # ------------------------------------------------------------------- #
    def coverage_range(self) -> tuple[int, int]:
        """Return ``(min_position, max_position_exclusive)`` covered by the cache.

        Returns ``(0, 0)`` if the manifest has no shards.
        """
        if not self._entries:
            return (0, 0)
        return (
            self._entries[0].start_token_position,
            self._entries[-1].end_token_position,
        )

    def has_coverage(self, start: int, end: int) -> bool:
        """True iff every position in ``[start, end)`` is in some shard."""
        try:
            self._manifest.assert_covers(start, end)
            return True
        except ValueError:
            return False

    # ------------------------------------------------------------------- #
    # Random-access read
    # ------------------------------------------------------------------- #
    def get_topk(self, positions: Any) -> tuple[Any, Any]:
        """For each position in ``positions[N]`` return ``(logits[N, K] float32, indices[N, K] uint32)``.

        ``positions`` must be 1-D integer-typed numpy array. Order of return
        matches order of input. Raises ``ValueError`` if any position lies
        outside the cache coverage.

        Logits are float32 in the return value even though the on-disk
        format stores bf16 bit patterns in uint16 — the conversion happens
        inside this method via ``_bf16_bits_to_f32``. Consumers can pass
        the returned array directly into a KL loss without further casting.
        """
        import numpy as np
        import pyarrow as pa

        positions = np.asarray(positions, dtype=np.int64)
        if positions.ndim != 1:
            raise ValueError(f"positions must be 1-D, got shape {positions.shape}")
        n = positions.shape[0]
        if n == 0:
            return (
                np.empty((0, self.top_k), dtype=np.float32),
                np.empty((0, self.top_k), dtype=np.uint32),
            )

        # Group positions by shard.
        shard_idx = np.empty(n, dtype=np.int64)
        for i, p in enumerate(positions):
            si = self._find_shard_index(int(p))
            if si is None:
                raise ValueError(
                    f"position {int(p)} is outside cache coverage "
                    f"{self.coverage_range()} for teacher {self.teacher_id!r}"
                )
            shard_idx[i] = si

        # Accumulate uint16 bf16-bits first (cheap), then bitcast → float32
        # once at the end. Doing it once avoids per-shard bitcast overhead.
        out_logits_u16 = np.empty((n, self.top_k), dtype=np.uint16)
        out_indices = np.empty((n, self.top_k), dtype=np.uint32)

        unique_shards = np.unique(shard_idx)
        for si in unique_shards:
            entry = self._entries[int(si)]
            table = self._open_shard(entry)
            # Row offsets within the shard.
            mask = shard_idx == si
            row_idx_in_shard = (positions[mask] - entry.start_token_position).astype(np.int64)

            # Take requested rows from each column (zero-copy slices).
            take = pa.array(row_idx_in_shard, type=pa.int64())
            logits_col = table.column("logits").take(take)
            indices_col = table.column("indices").take(take)

            # FixedSizeBinaryArray → uint16 view (still bf16 bit pattern).
            raw = b"".join(logits_col.to_pylist())
            logits_arr_u16 = np.frombuffer(raw, dtype=np.uint16).reshape(-1, self.top_k)
            out_logits_u16[mask] = logits_arr_u16

            indices_arr = np.array(indices_col.to_pylist(), dtype=np.uint32)
            out_indices[mask] = indices_arr

        # Reinterpret bf16 bit patterns as float32 at the boundary.
        # See _bf16_bits_to_f32 docstring for why this matters.
        out_logits = _bf16_bits_to_f32(out_logits_u16)
        return out_logits, out_indices

    # ------------------------------------------------------------------- #
    # Shard lookup + mmap LRU
    # ------------------------------------------------------------------- #
    def _find_shard_index(self, position: int) -> int | None:
        """Binary-search the entry index whose ``[start, end)`` covers ``position``."""
        if not self._starts:
            return None
        idx = bisect.bisect_right(self._starts, position) - 1
        if 0 <= idx < len(self._entries):
            entry = self._entries[idx]
            if entry.start_token_position <= position < entry.end_token_position:
                return idx
        return None

    def _open_shard(self, entry: ShardManifestEntry) -> Any:
        """Return the pyarrow Table for ``entry``, mmap-backed, LRU-cached."""
        import pyarrow as pa

        key = entry.r2_key
        if key in self._open:
            # Move to MRU position.
            value = self._open.pop(key)
            self._open[key] = value
            return value[1]

        local_path = compute_local_path(self.shard_root, key)
        if not local_path.exists():
            raise FileNotFoundError(
                f"shard for token range "
                f"[{entry.start_token_position}, {entry.end_token_position}) "
                f"not present locally at {local_path}. "
                f"R2 lazy-fetch is a follow-up; for now, pre-download shards."
            )
        if self.verify_sha256:
            actual = _sha256_file(local_path)
            if actual != entry.sha256:
                raise ValueError(
                    f"sha256 mismatch for {local_path}: "
                    f"expected {entry.sha256}, got {actual}"
                )

        mmap_source = pa.memory_map(str(local_path), "r")
        reader = pa.ipc.open_file(mmap_source)
        table = reader.read_all()
        # LRU eviction. Plain dicts since Python 3.7 preserve insertion
        # order, so iter(dict) yields oldest key first.
        while len(self._open) >= self.max_open_shards:
            oldest_key = next(iter(self._open))
            old_mmap, _ = self._open.pop(oldest_key)
            try:
                old_mmap.close()
            except (AttributeError, OSError):
                pass
        self._open[key] = (mmap_source, table)
        return table

    def close(self) -> None:
        """Close all mmap'd shards."""
        for _key, (mmap, _table) in list(self._open.items()):
            try:
                mmap.close()
            except (AttributeError, OSError):
                pass
        self._open.clear()


class MultiTeacherCacheReader:
    """Convenience wrapper that fans ``get_topk`` across multiple teachers.

    Returned arrays have a leading teacher axis: ``logits[T, N, K]``,
    ``indices[T, N, K]``. The teacher order matches the order of the
    ``teacher_ids`` passed to the constructor.
    """

    def __init__(self, readers: list[TeacherCacheReader]):
        if not readers:
            raise ValueError("MultiTeacherCacheReader needs >= 1 reader")
        k_values = {r.top_k for r in readers}
        if len(k_values) != 1:
            raise ValueError(
                f"all teachers must share the same top_k; got {k_values}"
            )
        self.readers = readers
        self.top_k = readers[0].top_k

    @property
    def teacher_ids(self) -> list[str]:
        return [r.teacher_id for r in self.readers]

    def get_topk(self, positions: Any) -> tuple[Any, Any]:
        """Stacked across teachers: returns ``(logits[T,N,K], indices[T,N,K])``."""
        import numpy as np

        per_teacher_logits = []
        per_teacher_indices = []
        for r in self.readers:
            lg, ix = r.get_topk(positions)
            per_teacher_logits.append(lg)
            per_teacher_indices.append(ix)
        return (
            np.stack(per_teacher_logits, axis=0),
            np.stack(per_teacher_indices, axis=0),
        )

    def close(self) -> None:
        for r in self.readers:
            r.close()
