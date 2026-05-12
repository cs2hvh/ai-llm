"""Offline packed-corpus storage: uint32 token shards + provenance.

B2 work from the 2026-05-12 reviewer Q&A (``docs/reviewer_qa_2026-05-12.md``
§4) and plan v3 (``docs/plan_v3_after_review3.md`` §4). This module is
the *data plane* — writer, reader, manifest, and seek index. The build
pipeline that pulls real HF data, dedupes, tokenizes, and feeds this
writer is a separate piece (``scripts/build_packed_corpus.py``).

Layout on disk::

    <corpus_root>/
        manifest.json                # CorpusManifest (top-level)
        shard-0000/
            tokens.bin               # uint32 little-endian, N × seq_len tokens
            seq_meta.arrow           # per-packed-sequence metadata (IPC stream)
            doc_meta.parquet         # per-doc-span provenance (dict-encoded)
            manifest.json            # ShardManifest
        shard-0001/
            ...

Why uint32 (NOT uint16):
    Our SentencePiece vocab is 131,072. uint16 only addresses 0-65,535;
    storing a token id ≥ 65,536 would silently wrap to a different
    token. This would corrupt half the vocabulary with no error signal.
    Always uint32 on disk. (See ``feedback_uint32_for_131k_vocab`` in
    the user's memory.)

Why a simple seek-index math, not a per-position table:
    Packed sequences are fixed-length. For any sequence_id::

        shard_id     = sequence_id // sequences_per_shard
        local_offset = sequence_id %  sequences_per_shard
        byte_offset  = local_offset * sequence_length * 4

    No table lookup. O(1), no I/O until the actual token read.

Why per-shard subdirectories (vs all files in one dir):
    Easier R2 prefix mirroring; cheap atomic rotation when one worker
    finishes a shard; readers can pre-mmap one shard at a time without
    accidentally opening 1,954 files.

The schemas (SequenceMeta, DocSpan, ShardManifest, CorpusManifest) are
locked per reviewer Q&A §4 — changes here are a format version bump
plus a write-old-read-new compat shim, not a silent schema drift.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from myllm.utils import get_logger

log = get_logger(__name__)

# Format version. Bump on any breaking schema change.
CORPUS_FORMAT_VERSION = 1

# Token storage dtype. uint32 is non-negotiable for 131k+ vocab.
TOKEN_DTYPE = np.uint32
TOKEN_BYTES = 4

# Default shard size per reviewer Q&A §4 — 512M tokens/shard ≈ 2GB.
# Pick something reasonable for tests/smoke; production builds override.
DEFAULT_SEQUENCES_PER_SHARD_HINT = 524_288_000


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DocSpan:
    """One source-document's contribution to a packed sequence.

    A packed sequence usually concatenates several documents separated by
    EOS tokens. ``DocSpan`` records the byte-of-origin so we can answer
    "which source did position N come from?" for distillation cache
    alignment, contamination reports, and source-level validation slicing.
    """

    doc_span_id: int           # unique within the corpus
    sequence_id: int           # back-reference to the SequenceMeta row
    source_id: str             # e.g. "fineweb-edu" — dictionary-encoded
    doc_id_hash: int           # xxhash64 of source-doc id (URL, file path, ...)
    dataset_revision_id: str   # HF commit SHA, dataset version tag, etc.
    token_start_in_sequence: int   # 0-indexed, inclusive
    token_end_in_sequence: int     # exclusive
    text_hash: int             # xxhash64 of the raw text of the originating doc


@dataclass(frozen=True)
class SequenceMeta:
    """Per-packed-sequence header. One row in seq_meta.arrow."""

    sequence_id: int                       # unique within corpus
    token_start_global: int                # global token offset (inclusive)
    token_end_global: int                  # global token offset (exclusive)
    source_mix: dict[str, int]             # source_id → tokens-from-source
    doc_span_start_id: int                 # first doc_span_id in doc_meta.parquet
    doc_span_count: int                    # number of doc spans in this sequence


@dataclass(frozen=True)
class ShardManifest:
    """One shard's manifest.json. Written last as the completion marker."""

    format_version: int
    shard_id: int
    sequence_length: int                   # tokens per packed sequence
    actual_sequences: int                  # may be < sequences_per_shard for last
    total_tokens_in_shard: int
    tokenizer_sha256: str
    source_mix_histogram: dict[str, int]   # aggregate token counts across all sequences
    build_timestamp_utc: str
    first_sequence_id: int                 # inclusive
    last_sequence_id: int                  # inclusive; -1 if shard is empty
    first_doc_span_id: int                 # inclusive
    doc_span_count: int                    # number of doc spans in this shard

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ShardManifest":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__})


@dataclass(frozen=True)
class CorpusManifest:
    """Top-level corpus manifest.json — written by the build orchestrator
    after all shards are complete, NOT by the writer.

    Carries everything a reader needs to validate compatibility before
    reading any tokens: tokenizer fingerprint, sequence length (to
    compute byte_offset), sequences_per_shard (for the seek index),
    source-share planned vs measured (so we can detect a mixer drift
    after the fact).
    """

    format_version: int
    corpus_name: str
    tokenizer_sha256: str
    sequence_length: int
    sequences_per_shard: int
    n_shards: int
    total_sequences: int
    total_tokens: int
    source_revisions: dict[str, str]            # source_id → HF commit / version
    target_source_share: dict[str, float]       # planned mix
    actual_source_share: dict[str, float]       # measured token share
    build_timestamp_utc: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CorpusManifest":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__})


# --------------------------------------------------------------------------- #
# Seek math (no class needed — these are pure functions used everywhere)
# --------------------------------------------------------------------------- #
def shard_id_for(sequence_id: int, sequences_per_shard: int) -> int:
    """Which shard contains ``sequence_id``."""
    if sequence_id < 0:
        raise ValueError(f"sequence_id must be >= 0, got {sequence_id}")
    if sequences_per_shard < 1:
        raise ValueError(f"sequences_per_shard must be >= 1, got {sequences_per_shard}")
    return sequence_id // sequences_per_shard


def local_offset_for(sequence_id: int, sequences_per_shard: int) -> int:
    """Position within the shard (0-indexed)."""
    if sequence_id < 0:
        raise ValueError(f"sequence_id must be >= 0, got {sequence_id}")
    return sequence_id % sequences_per_shard


def byte_offset_for(sequence_id: int, sequences_per_shard: int, sequence_length: int) -> int:
    """File-byte offset where the sequence starts in its shard's tokens.bin."""
    return local_offset_for(sequence_id, sequences_per_shard) * sequence_length * TOKEN_BYTES


def packed_sequence_bytes(sequence_length: int) -> int:
    """Bytes consumed by one packed sequence in tokens.bin."""
    return sequence_length * TOKEN_BYTES


# --------------------------------------------------------------------------- #
# Writer
# --------------------------------------------------------------------------- #
class PackedCorpusWriter:
    """Append-only writer that rotates shards as they fill.

    Single-shard-range usage (one worker, one shard range)::

        w = PackedCorpusWriter(
            root="/data/corpus_v1/sources/fineweb_edu",
            sequence_length=8192,
            sequences_per_shard=65536,
            tokenizer_sha256="deadbeef...",
            first_shard_id=0,
        )
        for tokens, doc_spans in packed_iter:
            w.append_sequence(tokens, doc_spans)
        shard_manifests = w.close()

    NOT thread-safe. Each worker should own a disjoint shard range. The
    top-level ``CorpusManifest`` is built by the launcher after all
    workers finish — it merges per-shard manifests.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        sequence_length: int,
        sequences_per_shard: int,
        tokenizer_sha256: str,
        first_shard_id: int = 0,
        first_sequence_id: int = 0,
        first_doc_span_id: int = 0,
        r2_prefix: str | None = None,
        delete_local_after_upload: bool = False,
    ):
        if sequence_length < 1:
            raise ValueError(f"sequence_length must be >= 1, got {sequence_length}")
        if sequences_per_shard < 1:
            raise ValueError(
                f"sequences_per_shard must be >= 1, got {sequences_per_shard}"
            )
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.sequence_length = int(sequence_length)
        self.sequences_per_shard = int(sequences_per_shard)
        self.tokenizer_sha256 = str(tokenizer_sha256)
        # R2 streaming mirror — when ``r2_prefix`` is set, each shard's
        # files are uploaded to ``s3://<bucket>/<r2_prefix>/shard-NNNNNN/*``
        # immediately after the shard closes. Optionally deletes the local
        # copy after a successful upload (lets multi-TB builds run on
        # small local disk by streaming to object storage).
        self.r2_prefix = r2_prefix.rstrip("/") if r2_prefix else None
        self.delete_local_after_upload = bool(delete_local_after_upload)

        # Counter state — advances as sequences are appended.
        self._shard_id = int(first_shard_id)
        self._next_sequence_id = int(first_sequence_id)
        self._next_doc_span_id = int(first_doc_span_id)

        # Current shard's in-memory staging buffers.
        # tokens.bin is written incrementally; metadata is staged in memory
        # and flushed at shard close (the volume is small relative to tokens).
        self._tokens_file = None
        self._seq_meta: list[SequenceMeta] = []
        self._doc_spans: list[DocSpan] = []
        self._shard_first_sequence_id: int | None = None
        self._shard_first_doc_span_id: int | None = None
        self._shard_total_tokens: int = 0
        self._shard_source_mix: dict[str, int] = {}

        self._closed_shards: list[ShardManifest] = []
        self._opened = False

    # ------------------------------------------------------------------- #
    # Per-shard lifecycle
    # ------------------------------------------------------------------- #
    def _shard_dir(self, shard_id: int) -> Path:
        return self.root / f"shard-{shard_id:06d}"

    def _open_current_shard(self) -> None:
        d = self._shard_dir(self._shard_id)
        d.mkdir(parents=True, exist_ok=True)
        self._tokens_file = open(d / "tokens.bin", "wb")
        self._shard_first_sequence_id = self._next_sequence_id
        self._shard_first_doc_span_id = self._next_doc_span_id
        self._shard_total_tokens = 0
        self._shard_source_mix = {}
        self._seq_meta = []
        self._doc_spans = []
        log.info(
            "packed_corpus_shard_open",
            shard_id=self._shard_id,
            path=str(d),
            first_sequence_id=self._next_sequence_id,
        )

    def _close_current_shard(self) -> ShardManifest:
        """Flush metadata + manifest for the current shard."""
        if self._tokens_file is None:
            raise RuntimeError("attempted to close shard but no shard is open")
        self._tokens_file.close()
        self._tokens_file = None

        d = self._shard_dir(self._shard_id)
        actual_sequences = len(self._seq_meta)

        # Write seq_meta.arrow + doc_meta.parquet (only if we have any data).
        if actual_sequences > 0:
            _write_seq_meta(d / "seq_meta.arrow", self._seq_meta)
            _write_doc_meta(d / "doc_meta.parquet", self._doc_spans)

        last_seq_id = (
            (self._shard_first_sequence_id + actual_sequences - 1)
            if actual_sequences > 0
            else -1
        )
        manifest = ShardManifest(
            format_version=CORPUS_FORMAT_VERSION,
            shard_id=self._shard_id,
            sequence_length=self.sequence_length,
            actual_sequences=actual_sequences,
            total_tokens_in_shard=self._shard_total_tokens,
            tokenizer_sha256=self.tokenizer_sha256,
            source_mix_histogram=dict(self._shard_source_mix),
            build_timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            first_sequence_id=(
                self._shard_first_sequence_id if actual_sequences > 0 else -1
            ),
            last_sequence_id=last_seq_id,
            first_doc_span_id=(
                self._shard_first_doc_span_id
                if actual_sequences > 0 and self._doc_spans
                else -1
            ),
            doc_span_count=len(self._doc_spans),
        )
        # manifest.json is written LAST as the completion marker. A reader
        # that sees tokens.bin without manifest.json knows the shard was
        # interrupted (matches the bug-#6 partial-write protection pattern).
        _write_json_atomic(d / "manifest.json", manifest.to_dict())
        log.info(
            "packed_corpus_shard_close",
            shard_id=self._shard_id,
            actual_sequences=actual_sequences,
            total_tokens=self._shard_total_tokens,
        )
        # Optional R2 streaming mirror — runs synchronously so a kill mid-
        # upload doesn't leave a half-uploaded shard that the reader could
        # treat as complete. (Upload errors don't roll back the local shard
        # — the operator can re-sync later.)
        if self.r2_prefix is not None:
            self._mirror_shard_to_r2(d)
        return manifest

    def _mirror_shard_to_r2(self, shard_dir: Path) -> None:
        """Upload all files in ``shard_dir`` to R2, optionally delete local."""
        try:
            from myllm.utils.storage import upload_directory
        except ImportError:
            log.warning(
                "packed_corpus_r2_mirror_skipped_storage_unavailable",
                shard=shard_dir.name,
            )
            return
        remote = f"{self.r2_prefix}/{shard_dir.name}"
        try:
            n = upload_directory(shard_dir, remote)
            log.info(
                "packed_corpus_shard_uploaded",
                shard=shard_dir.name,
                remote_prefix=remote,
                files=n,
            )
        except Exception as e:  # noqa: BLE001
            log.error(
                "packed_corpus_shard_upload_failed",
                shard=shard_dir.name,
                remote_prefix=remote,
                error=str(e),
            )
            return  # don't delete local on failure
        if self.delete_local_after_upload:
            import shutil
            shutil.rmtree(shard_dir, ignore_errors=True)
            log.info("packed_corpus_shard_local_deleted", shard=shard_dir.name)

    # ------------------------------------------------------------------- #
    # Append
    # ------------------------------------------------------------------- #
    def append_sequence(
        self, tokens: np.ndarray, doc_spans: list[DocSpan]
    ) -> int:
        """Append one packed sequence + its doc spans.

        ``tokens`` must be 1-D with length == ``sequence_length``, dtype
        coercible to uint32, all values in [0, 2**32). Returns the
        assigned ``sequence_id``.

        ``doc_spans`` describe which source-docs contributed which
        token ranges in this sequence. The caller is responsible for
        their semantic correctness; the writer enforces:
          - all spans have ``sequence_id`` field set later (we overwrite)
          - all spans have ``doc_span_id`` field set later (we assign)
          - token ranges within [0, sequence_length] (no enforcement of
            non-overlap; that's a packer responsibility)
        """
        if not self._opened:
            self._open_current_shard()
            self._opened = True

        # Validate token array.
        arr = np.asarray(tokens)
        if arr.ndim != 1:
            raise ValueError(f"tokens must be 1-D; got shape {arr.shape}")
        if arr.shape[0] != self.sequence_length:
            raise ValueError(
                f"tokens length {arr.shape[0]} != writer sequence_length "
                f"{self.sequence_length}"
            )
        if arr.dtype != TOKEN_DTYPE:
            # Coerce — but reject if any value can't fit in uint32.
            if np.issubdtype(arr.dtype, np.signedinteger) and (arr < 0).any():
                raise ValueError("tokens contain negative ids; cannot fit in uint32")
            arr = arr.astype(TOKEN_DTYPE, copy=False)

        assigned_sequence_id = self._next_sequence_id

        # Assign doc_span_ids + back-reference each span to this sequence.
        assigned_doc_spans: list[DocSpan] = []
        source_mix: dict[str, int] = {}
        for span in doc_spans:
            if span.token_end_in_sequence > self.sequence_length:
                raise ValueError(
                    f"doc span ends at {span.token_end_in_sequence} > "
                    f"sequence_length {self.sequence_length}"
                )
            if span.token_start_in_sequence < 0:
                raise ValueError(
                    f"doc span starts at {span.token_start_in_sequence} < 0"
                )
            new_span = DocSpan(
                doc_span_id=self._next_doc_span_id,
                sequence_id=assigned_sequence_id,
                source_id=span.source_id,
                doc_id_hash=span.doc_id_hash,
                dataset_revision_id=span.dataset_revision_id,
                token_start_in_sequence=span.token_start_in_sequence,
                token_end_in_sequence=span.token_end_in_sequence,
                text_hash=span.text_hash,
            )
            assigned_doc_spans.append(new_span)
            n_tokens = (
                span.token_end_in_sequence - span.token_start_in_sequence
            )
            source_mix[span.source_id] = source_mix.get(span.source_id, 0) + n_tokens
            self._next_doc_span_id += 1

        # Build SequenceMeta.
        token_start_global = (
            self._shard_first_sequence_id * self.sequence_length
            if self._shard_first_sequence_id is not None
            else 0
        )
        # NOTE: token_start_global is computed relative to the *writer's*
        # first_sequence_id, not a corpus-wide cursor. The launcher knows
        # the global offset of each worker's shard range and remaps when
        # building the CorpusManifest. For a single-writer corpus this
        # coincides with the absolute token offset.
        seq_token_start = assigned_sequence_id * self.sequence_length
        meta = SequenceMeta(
            sequence_id=assigned_sequence_id,
            token_start_global=seq_token_start,
            token_end_global=seq_token_start + self.sequence_length,
            source_mix=source_mix,
            doc_span_start_id=(
                assigned_doc_spans[0].doc_span_id if assigned_doc_spans else -1
            ),
            doc_span_count=len(assigned_doc_spans),
        )

        # Write tokens to tokens.bin.
        self._tokens_file.write(arr.tobytes(order="C"))
        self._shard_total_tokens += self.sequence_length

        # Stage metadata.
        self._seq_meta.append(meta)
        self._doc_spans.extend(assigned_doc_spans)
        for src, n in source_mix.items():
            self._shard_source_mix[src] = self._shard_source_mix.get(src, 0) + n

        self._next_sequence_id += 1

        # Rotate to next shard if we just filled this one.
        if len(self._seq_meta) >= self.sequences_per_shard:
            mf = self._close_current_shard()
            self._closed_shards.append(mf)
            self._shard_id += 1
            self._opened = False  # next append will reopen

        return assigned_sequence_id

    def close(self) -> list[ShardManifest]:
        """Flush the in-flight shard (if any) + return all per-shard manifests."""
        if self._opened:
            mf = self._close_current_shard()
            self._closed_shards.append(mf)
            self._opened = False
        return list(self._closed_shards)


# --------------------------------------------------------------------------- #
# Reader
# --------------------------------------------------------------------------- #
class PackedCorpusReader:
    """memmap-backed random-access reader.

    Built from a top-level ``CorpusManifest``. Lazily opens per-shard
    memmaps on first access; LRU-evicts when ``max_open_shards`` is
    exceeded. Suitable for training-loop sampling (sequence-id → tokens).

    Thread-safety: NOT thread-safe (numpy memmap is). Each worker should
    construct its own reader.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        max_open_shards: int = 8,
    ):
        self.root = Path(root).resolve()
        manifest_path = self.root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"corpus manifest not found at {manifest_path}; "
                "did the build complete? (manifest.json is written last)"
            )
        self.manifest = CorpusManifest.from_dict(_read_json(manifest_path))
        if self.manifest.format_version != CORPUS_FORMAT_VERSION:
            raise ValueError(
                f"corpus format_version {self.manifest.format_version} != "
                f"reader version {CORPUS_FORMAT_VERSION}"
            )
        self.max_open_shards = int(max_open_shards)
        # LRU of opened shards: shard_id → np.memmap of tokens.bin
        self._open_token_maps: dict[int, np.memmap] = {}
        # LRU of opened seq_meta tables: shard_id → list[SequenceMeta]
        self._open_seq_meta: dict[int, list[SequenceMeta]] = {}
        # LRU of opened doc_meta tables: shard_id → list[DocSpan]
        self._open_doc_meta: dict[int, list[DocSpan]] = {}

    # ------------------------------------------------------------------- #
    # Shape queries
    # ------------------------------------------------------------------- #
    @property
    def total_sequences(self) -> int:
        return self.manifest.total_sequences

    @property
    def total_tokens(self) -> int:
        return self.manifest.total_tokens

    @property
    def sequence_length(self) -> int:
        return self.manifest.sequence_length

    @property
    def sequences_per_shard(self) -> int:
        return self.manifest.sequences_per_shard

    # ------------------------------------------------------------------- #
    # Sequence access
    # ------------------------------------------------------------------- #
    def get_sequence(self, sequence_id: int) -> np.ndarray:
        """Return the uint32 token array for ``sequence_id``.

        The returned array is a memmap view (not a copy). Callers that
        intend to mutate or hold references across many reads should
        call ``np.array(...)`` to copy.
        """
        if sequence_id < 0 or sequence_id >= self.total_sequences:
            raise IndexError(
                f"sequence_id {sequence_id} out of range "
                f"[0, {self.total_sequences})"
            )
        shard_id = shard_id_for(sequence_id, self.sequences_per_shard)
        local = local_offset_for(sequence_id, self.sequences_per_shard)
        mm = self._open_tokens_mmap(shard_id)
        start = local * self.sequence_length
        end = start + self.sequence_length
        return mm[start:end]

    def iterate_from(
        self, start_sequence_id: int = 0
    ) -> Iterator[tuple[int, np.ndarray]]:
        """Yield ``(sequence_id, tokens)`` from start_sequence_id onward."""
        if start_sequence_id < 0:
            raise ValueError(f"start_sequence_id must be >= 0, got {start_sequence_id}")
        for sid in range(start_sequence_id, self.total_sequences):
            yield sid, self.get_sequence(sid)

    # ------------------------------------------------------------------- #
    # Provenance access
    # ------------------------------------------------------------------- #
    def get_sequence_meta(self, sequence_id: int) -> SequenceMeta:
        shard_id = shard_id_for(sequence_id, self.sequences_per_shard)
        local = local_offset_for(sequence_id, self.sequences_per_shard)
        metas = self._open_seq_meta_table(shard_id)
        if local >= len(metas):
            raise IndexError(
                f"sequence_id {sequence_id} not in shard {shard_id} "
                f"(only {len(metas)} sequences in shard)"
            )
        return metas[local]

    def get_provenance(self, sequence_id: int) -> list[DocSpan]:
        """Return all DocSpan rows whose ``sequence_id`` matches."""
        meta = self.get_sequence_meta(sequence_id)
        if meta.doc_span_count == 0:
            return []
        shard_id = shard_id_for(sequence_id, self.sequences_per_shard)
        spans = self._open_doc_meta_table(shard_id)
        # doc_span_start_id is absolute, but the spans list is per-shard.
        # Find the matching window by sequence_id (clean + robust).
        return [s for s in spans if s.sequence_id == sequence_id]

    def get_segment_ids(self, sequence_id: int) -> np.ndarray:
        """Reconstruct segment_ids for ``sequence_id`` from its DocSpan rows.

        Each DocSpan in the sequence becomes one segment (0, 1, 2, ...).
        Token positions not covered by any DocSpan get segment_id = -1,
        the sentinel used by ``make_input_label_pairs`` to zero-out loss
        at document boundaries and padding positions.

        This matches the semantics of the original SequencePacker's
        ``segment_ids`` output, where:
          - EOS token between docs belongs to the doc it terminates
            (same segment_id as preceding content)
          - Padding positions get segment_id = -1

        Returns a 1-D int32 array of length ``sequence_length``.
        """
        spans = self.get_provenance(sequence_id)
        segment_ids = np.full(self.sequence_length, -1, dtype=np.int32)
        # Spans are NOT guaranteed sorted by start in the parquet; sort here
        # so segment_ids increase left-to-right.
        sorted_spans = sorted(spans, key=lambda s: s.token_start_in_sequence)
        for seg_idx, span in enumerate(sorted_spans):
            start = max(0, span.token_start_in_sequence)
            end = min(self.sequence_length, span.token_end_in_sequence)
            if end > start:
                segment_ids[start:end] = seg_idx
        return segment_ids

    def get_sequence_and_segments(
        self, sequence_id: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convenience: ``(tokens uint32, segment_ids int32)`` for one sequence."""
        tokens = self.get_sequence(sequence_id)
        seg_ids = self.get_segment_ids(sequence_id)
        return tokens, seg_ids

    def shard_manifest(self, shard_id: int) -> ShardManifest:
        path = self.root / f"shard-{shard_id:06d}" / "manifest.json"
        if not path.exists():
            raise FileNotFoundError(f"shard manifest missing: {path}")
        return ShardManifest.from_dict(_read_json(path))

    # ------------------------------------------------------------------- #
    # Shard-level open (LRU)
    # ------------------------------------------------------------------- #
    def _evict_if_full(self, table: dict[int, Any]) -> None:
        while len(table) >= self.max_open_shards:
            oldest = next(iter(table))
            del table[oldest]

    def _open_tokens_mmap(self, shard_id: int) -> np.memmap:
        if shard_id in self._open_token_maps:
            return self._open_token_maps[shard_id]
        self._evict_if_full(self._open_token_maps)
        path = self.root / f"shard-{shard_id:06d}" / "tokens.bin"
        if not path.exists():
            raise FileNotFoundError(f"shard tokens.bin missing: {path}")
        mm = np.memmap(path, dtype=TOKEN_DTYPE, mode="r")
        self._open_token_maps[shard_id] = mm
        return mm

    def _open_seq_meta_table(self, shard_id: int) -> list[SequenceMeta]:
        if shard_id in self._open_seq_meta:
            return self._open_seq_meta[shard_id]
        self._evict_if_full(self._open_seq_meta)
        path = self.root / f"shard-{shard_id:06d}" / "seq_meta.arrow"
        if not path.exists():
            raise FileNotFoundError(f"shard seq_meta.arrow missing: {path}")
        metas = _read_seq_meta(path)
        self._open_seq_meta[shard_id] = metas
        return metas

    def _open_doc_meta_table(self, shard_id: int) -> list[DocSpan]:
        if shard_id in self._open_doc_meta:
            return self._open_doc_meta[shard_id]
        self._evict_if_full(self._open_doc_meta)
        path = self.root / f"shard-{shard_id:06d}" / "doc_meta.parquet"
        if not path.exists():
            # An empty shard can legitimately have no doc_meta.parquet.
            self._open_doc_meta[shard_id] = []
            return []
        spans = _read_doc_meta(path)
        self._open_doc_meta[shard_id] = spans
        return spans


# --------------------------------------------------------------------------- #
# Top-level manifest helpers — used by the build orchestrator (not the writer)
# --------------------------------------------------------------------------- #
def write_corpus_manifest(
    root: str | Path,
    *,
    corpus_name: str,
    tokenizer_sha256: str,
    sequence_length: int,
    sequences_per_shard: int,
    source_revisions: dict[str, str],
    target_source_share: dict[str, float],
    shard_manifests: list[ShardManifest] | None = None,
) -> CorpusManifest:
    """Aggregate per-shard manifests, write the top-level manifest.json.

    Two modes:
      - ``shard_manifests=None`` (default): walks ``root/shard-*/manifest.json``
        on the local filesystem. The original mode — works when all shards
        are still on disk.
      - ``shard_manifests=[...]``: uses the in-memory list directly. Required
        when ``PackedCorpusWriter`` ran with ``delete_local_after_upload=True``
        (the shard-* dirs are gone from local; the writer's ``close()`` return
        value carries the manifests).

    Always writes ``<root>/manifest.json`` to local disk. The caller is
    responsible for mirroring it to R2 (or letting the writer-side
    streaming-mirror handle it on the next shard close — but the top-level
    manifest is written AFTER the last shard, so it needs an explicit upload).
    """
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    if shard_manifests is None:
        shard_dirs = sorted(p for p in root.glob("shard-*") if p.is_dir())
        if not shard_dirs:
            raise ValueError(
                f"no shard-* directories found under {root} (and no "
                f"shard_manifests passed in; if you used "
                f"delete_local_after_upload=True, pass writer.close()'s "
                f"return value as shard_manifests=)"
            )
        loaded: list[ShardManifest] = []
        for d in shard_dirs:
            mp = d / "manifest.json"
            if not mp.exists():
                log.warning("packed_corpus_skipping_incomplete_shard", path=str(d))
                continue
            loaded.append(ShardManifest.from_dict(_read_json(mp)))
        shard_manifests = loaded

    if not shard_manifests:
        raise ValueError("no shard manifests found / passed; cannot aggregate")

    n_shards = 0
    total_sequences = 0
    total_tokens = 0
    actual_mix: dict[str, int] = {}
    for sm in shard_manifests:
        if sm.sequence_length != sequence_length:
            raise ValueError(
                f"shard {sm.shard_id} sequence_length {sm.sequence_length} "
                f"!= corpus sequence_length {sequence_length}"
            )
        if sm.tokenizer_sha256 != tokenizer_sha256:
            raise ValueError(
                f"shard {sm.shard_id} tokenizer_sha256 mismatch: "
                f"{sm.tokenizer_sha256} != {tokenizer_sha256}"
            )
        n_shards += 1
        total_sequences += sm.actual_sequences
        total_tokens += sm.total_tokens_in_shard
        for src, n in sm.source_mix_histogram.items():
            actual_mix[src] = actual_mix.get(src, 0) + n

    actual_share: dict[str, float] = {}
    if total_tokens > 0:
        actual_share = {k: v / total_tokens for k, v in actual_mix.items()}

    manifest = CorpusManifest(
        format_version=CORPUS_FORMAT_VERSION,
        corpus_name=str(corpus_name),
        tokenizer_sha256=str(tokenizer_sha256),
        sequence_length=int(sequence_length),
        sequences_per_shard=int(sequences_per_shard),
        n_shards=n_shards,
        total_sequences=total_sequences,
        total_tokens=total_tokens,
        source_revisions=dict(source_revisions),
        target_source_share=dict(target_source_share),
        actual_source_share=actual_share,
        build_timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    _write_json_atomic(root / "manifest.json", manifest.to_dict())
    log.info(
        "packed_corpus_manifest_written",
        root=str(root),
        n_shards=n_shards,
        total_sequences=total_sequences,
        total_tokens=total_tokens,
    )
    return manifest


# --------------------------------------------------------------------------- #
# Training-loop adapter — bridges PackedCorpusReader to the existing
# ``make_input_label_pairs`` 4-tuple shape that the training loop consumes.
# --------------------------------------------------------------------------- #
def iter_packed_pairs(
    reader: PackedCorpusReader,
    *,
    start_sequence_id: int = 0,
) -> Iterator[tuple[list[int], list[int], list[int], list[int]]]:
    """Yield ``(input_ids, labels, segment_ids, loss_mask)`` tuples from a
    packed corpus — same shape as ``myllm.data.tokenize.make_input_label_pairs``.

    For each packed sequence:
      - input_ids = tokens[:-1]
      - labels    = tokens[1:]
      - segment_ids = segment_ids[:-1] (one per INPUT position)
      - loss_mask = 1 where the input and the label share a segment
        (so the next-token prediction is in-document) AND neither is -1
        (padding); 0 at doc boundaries and padding.

    ``start_sequence_id`` is the resume cursor. Combined with the training
    state's persisted ``data_position``, exact resume is:
        start_sequence_id = state["data_position"] // sequence_length
    The packed corpus is canonical (sequence_id → tokens never changes),
    so this gives bitwise-exact resume.
    """
    sid = start_sequence_id
    while sid < reader.total_sequences:
        tokens = reader.get_sequence(sid)
        seg_ids = reader.get_segment_ids(sid)
        if tokens.shape[0] < 2:
            sid += 1
            continue
        # int conversion — the training loop's batch_pairs expects Python
        # int / list, not numpy scalars (some downstream code is dtype-strict).
        token_list = [int(t) for t in tokens]
        seg_list = [int(s) for s in seg_ids]
        input_ids = token_list[:-1]
        labels = token_list[1:]
        input_segments = seg_list[:-1]
        label_segments = seg_list[1:]
        loss_mask = [
            1 if (a == b and a != -1) else 0
            for a, b in zip(input_segments, label_segments, strict=False)
        ]
        yield input_ids, labels, input_segments, loss_mask
        sid += 1


def sequence_id_from_data_position(data_position: int, sequence_length: int) -> int:
    """Convert a training-state ``data_position`` (in tokens) to the
    matching packed-corpus sequence_id resume cursor.

    Bitwise-exact resume: as long as the packed corpus has not been
    re-built since the checkpoint was written, ``start_sequence_id``
    fed to ``iter_packed_pairs`` recovers the exact token sequence
    the trainer was about to consume.
    """
    if sequence_length < 1:
        raise ValueError(f"sequence_length must be >= 1, got {sequence_length}")
    return int(data_position) // int(sequence_length)


def peek_data_position_from_checkpoint(checkpoint_root: str | Path) -> int:
    """Return the persisted ``data_position`` from the latest complete
    checkpoint under ``checkpoint_root``, or 0 if no checkpoint exists.

    Cheap: reads only the per-step ``manifest.json`` (small JSON file),
    never opens Orbax. Used by the packed-corpus data path to compute
    its resume ``start_sequence_id`` before constructing the iterator.

    The loop now writes ``data_position`` into the checkpoint manifest's
    ``extra`` block on every save (loop.py); older checkpoints predating
    that change will return 0 with no warning here — the caller can log
    if it wants.
    """
    root = Path(checkpoint_root)
    if not root.exists():
        return 0
    candidates: list[tuple[int, Path]] = []
    for d in sorted(root.glob("step-*")):
        manifest = d / "manifest.json"
        if not manifest.exists():
            continue
        try:
            m = _read_json(manifest)
            candidates.append((int(m["step"]), manifest))
        except (ValueError, KeyError, OSError):
            continue
    if not candidates:
        return 0
    # Latest by step.
    candidates.sort(key=lambda x: x[0])
    latest_manifest = _read_json(candidates[-1][1])
    extra = latest_manifest.get("extra", {}) or {}
    return int(extra.get("data_position", 0))


# --------------------------------------------------------------------------- #
# Arrow / Parquet I/O — kept private; schema lives here
# --------------------------------------------------------------------------- #
def _write_seq_meta(path: Path, metas: list[SequenceMeta]) -> None:
    """Serialize SequenceMeta list to an Arrow IPC file.

    Schema:
      sequence_id           int64
      token_start_global    int64
      token_end_global      int64
      source_mix_keys       list<string>     -- parallel to values; preserves dict
      source_mix_values     list<int64>
      doc_span_start_id     int64
      doc_span_count        int32
    """
    import pyarrow as pa
    import pyarrow.ipc as ipc

    keys = [list(m.source_mix.keys()) for m in metas]
    vals = [list(m.source_mix.values()) for m in metas]
    arrays = [
        pa.array([m.sequence_id for m in metas], type=pa.int64()),
        pa.array([m.token_start_global for m in metas], type=pa.int64()),
        pa.array([m.token_end_global for m in metas], type=pa.int64()),
        pa.array(keys, type=pa.list_(pa.string())),
        pa.array(vals, type=pa.list_(pa.int64())),
        pa.array([m.doc_span_start_id for m in metas], type=pa.int64()),
        pa.array([m.doc_span_count for m in metas], type=pa.int32()),
    ]
    table = pa.table(
        arrays,
        names=[
            "sequence_id",
            "token_start_global",
            "token_end_global",
            "source_mix_keys",
            "source_mix_values",
            "doc_span_start_id",
            "doc_span_count",
        ],
    )
    tmp = path.with_suffix(path.suffix + ".tmp")
    with ipc.new_file(tmp, table.schema) as w:
        w.write_table(table)
    tmp.replace(path)


def _read_seq_meta(path: Path) -> list[SequenceMeta]:
    import pyarrow.ipc as ipc

    with ipc.open_file(path) as r:
        table = r.read_all()
    metas: list[SequenceMeta] = []
    seq_ids = table.column("sequence_id").to_pylist()
    starts = table.column("token_start_global").to_pylist()
    ends = table.column("token_end_global").to_pylist()
    keys = table.column("source_mix_keys").to_pylist()
    vals = table.column("source_mix_values").to_pylist()
    ds_start = table.column("doc_span_start_id").to_pylist()
    ds_count = table.column("doc_span_count").to_pylist()
    for i in range(table.num_rows):
        source_mix = dict(zip(keys[i] or [], vals[i] or [], strict=False))
        metas.append(SequenceMeta(
            sequence_id=int(seq_ids[i]),
            token_start_global=int(starts[i]),
            token_end_global=int(ends[i]),
            source_mix=source_mix,
            doc_span_start_id=int(ds_start[i]),
            doc_span_count=int(ds_count[i]),
        ))
    return metas


def _write_doc_meta(path: Path, spans: list[DocSpan]) -> None:
    """Serialize DocSpan list to a Parquet file with dictionary encoding
    on the high-cardinality string columns.

    Schema:
      doc_span_id              int64
      sequence_id              int64
      source_id                string (dict-encoded)
      doc_id_hash              uint64
      dataset_revision_id      string (dict-encoded)
      token_start_in_sequence  int32
      token_end_in_sequence    int32
      text_hash                uint64
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not spans:
        # Even empty parquet write requires a schema. We never call this
        # for empty spans (writer skips when actual_sequences == 0); guard
        # anyway.
        return

    table = pa.table(
        {
            "doc_span_id": pa.array([s.doc_span_id for s in spans], type=pa.int64()),
            "sequence_id": pa.array([s.sequence_id for s in spans], type=pa.int64()),
            "source_id": pa.array([s.source_id for s in spans], type=pa.string()),
            "doc_id_hash": pa.array(
                [s.doc_id_hash for s in spans], type=pa.uint64()
            ),
            "dataset_revision_id": pa.array(
                [s.dataset_revision_id for s in spans], type=pa.string()
            ),
            "token_start_in_sequence": pa.array(
                [s.token_start_in_sequence for s in spans], type=pa.int32()
            ),
            "token_end_in_sequence": pa.array(
                [s.token_end_in_sequence for s in spans], type=pa.int32()
            ),
            "text_hash": pa.array(
                [s.text_hash for s in spans], type=pa.uint64()
            ),
        }
    )
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(
        table,
        tmp,
        compression="zstd",
        use_dictionary=["source_id", "dataset_revision_id"],
    )
    tmp.replace(path)


def _read_doc_meta(path: Path) -> list[DocSpan]:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    n = table.num_rows
    doc_span_id = table.column("doc_span_id").to_pylist()
    sequence_id = table.column("sequence_id").to_pylist()
    source_id = table.column("source_id").to_pylist()
    doc_id_hash = table.column("doc_id_hash").to_pylist()
    dataset_revision_id = table.column("dataset_revision_id").to_pylist()
    token_start = table.column("token_start_in_sequence").to_pylist()
    token_end = table.column("token_end_in_sequence").to_pylist()
    text_hash = table.column("text_hash").to_pylist()
    return [
        DocSpan(
            doc_span_id=int(doc_span_id[i]),
            sequence_id=int(sequence_id[i]),
            source_id=str(source_id[i]),
            doc_id_hash=int(doc_id_hash[i]),
            dataset_revision_id=str(dataset_revision_id[i]),
            token_start_in_sequence=int(token_start[i]),
            token_end_in_sequence=int(token_end[i]),
            text_hash=int(text_hash[i]),
        )
        for i in range(n)
    ]


# --------------------------------------------------------------------------- #
# JSON I/O — atomic write so a crash mid-flush can't leave a partial manifest
# --------------------------------------------------------------------------- #
def _write_json_atomic(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, sort_keys=True, indent=2))
    tmp.replace(path)


def _read_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)
