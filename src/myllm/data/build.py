"""Per-source build pipeline: HF stream → filter → dedupe → tokenize → pack → write.

Second half of B2 (data plane is in ``packed_corpus.py``). This module
chains the existing per-doc pipeline pieces (``loader``, ``filters``,
``decontamination``, ``dedupe``, ``tokenize``) into a single
``build_one_source`` orchestrator that produces a per-source packed
corpus on disk.

Per the reviewer Q&A §4 + plan v3 §4:
  - **Per-source** dedup, not global (FineWeb's explicit finding: global
    dedup is a quality trap). Each source gets its own MinHash index.
  - **Revision pinning** per source — the HF commit SHA / version tag is
    recorded in the manifest so the corpus is reproducible.
  - **Provenance preserved end-to-end**: every token's (source_id,
    doc_id_hash, dataset_revision_id, text_hash) is captured in DocSpan
    rows attached to its packed sequence.

The build runs one source at a time. Multi-source mixing happens in a
separate "compose" pass that reads N per-source corpora and emits a
mixed-training corpus enforcing the target token shares. That pass is
NOT in this module — it's the next piece.
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import xxhash

from myllm.data.dedupe import Deduplicator, MinHashConfig
from myllm.data.packed_corpus import (
    DocSpan,
    PackedCorpusWriter,
    TOKEN_DTYPE,
    write_corpus_manifest,
)
from myllm.data.types import Document
from myllm.utils import get_logger
from myllm.utils.exceptions import DataPipelineError

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Provenance-aware packing
# --------------------------------------------------------------------------- #
@dataclass
class _PendingDoc:
    """One document's tokens + provenance, staged in the packing buffer."""
    source_id: str
    doc_id_hash: int
    dataset_revision_id: str
    text_hash: int
    buf_start: int   # absolute position in the running buffer (inclusive)
    buf_end: int     # absolute position in the running buffer (exclusive)


def pack_with_provenance(
    docs: Iterable[tuple[str, str, list[int], int, str]],
    *,
    sequence_length: int,
    eos_token_id: int,
    pad_token_id: int = 0,
    drop_last: bool = True,
) -> Iterator[tuple[np.ndarray, list[DocSpan]]]:
    """Pack documents into fixed-length sequences with provenance tracking.

    Each input tuple is ``(doc_id, source_id, tokens, text_hash, revision_id)``.
    Documents are concatenated with an EOS separator and chunked into
    ``sequence_length``-token packed sequences.

    For each emitted packed sequence, the second return value is the
    list of DocSpan records describing which (source_id, doc_id_hash,
    revision_id, text_hash) contributed which ``token_start..token_end``
    range *within the packed sequence*.

    A document that spans a sequence boundary produces TWO DocSpan rows
    (one in each sequence) so per-sequence provenance is complete.

    Yields ``(tokens: np.uint32 array of length sequence_length,
              doc_spans: list[DocSpan])``.

    The DocSpan rows have ``doc_span_id = -1`` and ``sequence_id = -1``
    — the writer assigns those when ingesting.
    """
    if sequence_length < 2:
        raise DataPipelineError(f"sequence_length must be >= 2, got {sequence_length}")

    buf: list[int] = []
    pending: list[_PendingDoc] = []
    bytes_consumed = 0  # tokens permanently emitted (for buf-absolute → seq-relative math)

    for doc_id, source_id, tokens, text_hash, revision_id in docs:
        if not tokens:
            continue
        doc_id_hash = xxhash.xxh64(doc_id.encode("utf-8")).intdigest() & ((1 << 63) - 1)
        # Absolute start position in the buffer (which is bytes_consumed + len(buf)
        # tokens from the global stream — but we only ever use the relative
        # position when emitting a sequence).
        doc_buf_start = len(buf)
        for t in tokens:
            buf.append(int(t))
        # EOS belongs to the same doc as the content it terminates.
        buf.append(int(eos_token_id))
        doc_buf_end = len(buf)
        pending.append(_PendingDoc(
            source_id=source_id,
            doc_id_hash=doc_id_hash,
            dataset_revision_id=revision_id,
            text_hash=text_hash,
            buf_start=doc_buf_start,
            buf_end=doc_buf_end,
        ))

        # Emit packed sequences while the buffer has enough.
        while len(buf) >= sequence_length:
            tokens_out = np.array(buf[:sequence_length], dtype=TOKEN_DTYPE)
            # Find docs overlapping [0, sequence_length).
            spans: list[DocSpan] = []
            new_pending: list[_PendingDoc] = []
            for d in pending:
                if d.buf_end <= 0 or d.buf_start >= sequence_length:
                    # Fully outside this sequence (only happens for docs
                    # that ended before, which we should have already
                    # dropped — defensive guard).
                    if d.buf_start >= sequence_length:
                        new_pending.append(_PendingDoc(
                            source_id=d.source_id,
                            doc_id_hash=d.doc_id_hash,
                            dataset_revision_id=d.dataset_revision_id,
                            text_hash=d.text_hash,
                            buf_start=d.buf_start - sequence_length,
                            buf_end=d.buf_end - sequence_length,
                        ))
                    continue
                clipped_start = max(0, d.buf_start)
                clipped_end = min(sequence_length, d.buf_end)
                if clipped_end > clipped_start:
                    spans.append(DocSpan(
                        doc_span_id=-1,
                        sequence_id=-1,
                        source_id=d.source_id,
                        doc_id_hash=d.doc_id_hash,
                        dataset_revision_id=d.dataset_revision_id,
                        token_start_in_sequence=clipped_start,
                        token_end_in_sequence=clipped_end,
                        text_hash=d.text_hash,
                    ))
                # Carry over if doc extends past this sequence's end.
                if d.buf_end > sequence_length:
                    new_pending.append(_PendingDoc(
                        source_id=d.source_id,
                        doc_id_hash=d.doc_id_hash,
                        dataset_revision_id=d.dataset_revision_id,
                        text_hash=d.text_hash,
                        buf_start=0,  # starts at beginning of next sequence
                        buf_end=d.buf_end - sequence_length,
                    ))

            yield tokens_out, spans
            buf = buf[sequence_length:]
            pending = new_pending
            bytes_consumed += sequence_length

    # Trailing partial sequence — pad and emit if requested.
    if buf and not drop_last:
        pad_count = sequence_length - len(buf)
        tokens_out = np.array(buf + [int(pad_token_id)] * pad_count, dtype=TOKEN_DTYPE)
        spans: list[DocSpan] = []
        for d in pending:
            if d.buf_end <= 0:
                continue
            clipped_start = max(0, d.buf_start)
            clipped_end = min(len(buf), d.buf_end)  # don't include padding
            if clipped_end > clipped_start:
                spans.append(DocSpan(
                    doc_span_id=-1,
                    sequence_id=-1,
                    source_id=d.source_id,
                    doc_id_hash=d.doc_id_hash,
                    dataset_revision_id=d.dataset_revision_id,
                    token_start_in_sequence=clipped_start,
                    token_end_in_sequence=clipped_end,
                    text_hash=d.text_hash,
                ))
        yield tokens_out, spans


# --------------------------------------------------------------------------- #
# Stage helpers
# --------------------------------------------------------------------------- #
def _text_hash(text: str) -> int:
    """xxhash64 of UTF-8 bytes, truncated to 63 bits (parquet int64 friendly)."""
    return xxhash.xxh64(text.encode("utf-8")).intdigest() & ((1 << 63) - 1)


def _tokenizer_sha256(tokenizer_path: str | Path) -> str:
    """SHA256 of the tokenizer.json file. The manifest stores this so a
    reader can refuse to mix tokens from different tokenizers."""
    p = Path(tokenizer_path)
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
@dataclass
class BuildStats:
    """Counters returned by build_one_source for operator visibility."""
    docs_seen: int = 0
    docs_filtered: int = 0
    docs_contaminated: int = 0
    docs_deduped: int = 0
    docs_kept: int = 0
    tokens_emitted: int = 0
    sequences_emitted: int = 0
    # Production-mix target share for this source (read from pretrain_mix.yaml).
    # Per-source corpus's WITHIN-CORPUS target is always 1.0 (the corpus is
    # 100% one source by construction). This field is the planned share
    # *within the composed training mix* and is informational only.
    production_target_share: float = 0.0


# Default MinHash config — FineWeb/DataTrove deployed values.
# (112 perms ÷ 14 bands = 8 rows/band; ~0.75 Jaccard threshold.)
FINEWEB_MINHASH_CONFIG = MinHashConfig(
    num_perm=112,
    num_bands=14,
    ngram_size=5,
    threshold=0.75,
)


def build_one_source(
    *,
    source_id: str,
    docs: Iterable[Document],
    tokenizer: Any,
    output_dir: str | Path,
    sequence_length: int,
    sequences_per_shard: int,
    revision_id: str,
    tokenizer_sha256: str,
    eos_token_id: int,
    pad_token_id: int = 0,
    filter_fn: Callable[[Document], bool] | None = None,
    decontaminator: Any | None = None,
    dedupe_config: MinHashConfig | None = None,
    target_share: float = 1.0,
    drop_last: bool = True,
    sample_limit: int | None = None,
    tokenize_batch_size: int = 1000,
    r2_prefix: str | None = None,
    delete_local_after_upload: bool = False,
) -> tuple[BuildStats, Any]:
    """Build one per-source packed corpus on disk.

    Returns ``(stats, corpus_manifest)``.

    The build is **independent per source** — workers can run this in
    parallel over disjoint sources without coordination. Cross-source
    mixing is a separate pass (the launcher composes per-source corpora
    into a training-time mixed corpus).

    Pipeline (each stage is optional via None):
      1. Document stream (caller provides; typically from HFStreamLoader)
      2. ``filter_fn(doc) -> bool`` (length/repetition/PII/etc.)
      3. ``decontaminator.scan_document(doc.text)`` (benchmark n-gram filter)
      4. MinHash + LSH dedupe (default: FineWeb 112x14 config)
      5. Tokenizer (HF tokenizers.Tokenizer — caller passes the loaded one)
      6. ``pack_with_provenance`` → uint32 packed sequences + DocSpans
      7. ``PackedCorpusWriter`` → shard rotation + manifest

    Dedupe is per-source (not global) per FineWeb finding. Set
    ``dedupe_config=None`` to disable entirely (useful for smoke tests
    or sources where dedup has already been done upstream).
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = BuildStats()

    dedup = Deduplicator(dedupe_config) if dedupe_config is not None else None

    writer = PackedCorpusWriter(
        output_dir,
        sequence_length=sequence_length,
        sequences_per_shard=sequences_per_shard,
        tokenizer_sha256=tokenizer_sha256,
        r2_prefix=r2_prefix,
        delete_local_after_upload=delete_local_after_upload,
    )

    log.info(
        "build_one_source_start",
        source_id=source_id,
        output_dir=str(output_dir),
        sequence_length=sequence_length,
        sequences_per_shard=sequences_per_shard,
        revision_id=revision_id,
        dedupe_enabled=dedup is not None,
    )

    def _filtered_docs() -> Iterator[Document]:
        """Pre-tokenization filtering stages (cheap, single-threaded).

        Splits filter/decontaminate/dedupe (all O(text)) from tokenization
        (the only O(tokens) stage that benefits from batching). Documents
        surviving these stages flow into the batched tokenizer below.
        """
        for doc in docs:
            stats.docs_seen += 1
            if sample_limit is not None and stats.docs_kept + (
                # account for docs already accepted in the current pending batch
                len(_pending_batch)
            ) >= sample_limit:
                return
            if filter_fn is not None and not filter_fn(doc):
                stats.docs_filtered += 1
                continue
            if decontaminator is not None:
                matches = decontaminator.scan_document(doc.text)
                if matches:
                    stats.docs_contaminated += 1
                    continue
            if dedup is not None:
                stable_id = f"{source_id}::{doc.doc_id}"
                added, _ = dedup.add_if_new(stable_id, doc.text)
                if not added:
                    stats.docs_deduped += 1
                    continue
            yield doc

    # Pending-batch buffer for tokenizer.encode_batch — shared with the
    # filter-stage generator so sample_limit accounting is accurate.
    _pending_batch: list[Document] = []

    def _kept_doc_stream() -> Iterator[tuple[str, str, list[int], int, str]]:
        """Yield (doc_id, source_id, token_ids, text_hash, revision_id) tuples.

        Tokenization is BATCHED: we accumulate up to ``tokenize_batch_size``
        filtered docs, then call ``tokenizer.encode_batch`` which uses the
        HF tokenizers Rust core's internal Rayon parallelism across CPU cores.

        Single-doc ``tokenizer.encode`` only saturates one core; batch mode
        gets us closer to ~10M tok/sec aggregate on a 100+ core machine vs
        ~100K tok/sec single-threaded. ~100× speedup for the bottleneck stage.
        """
        def _flush(batch: list[Document]) -> Iterator[tuple[str, str, list[int], int, str]]:
            if not batch:
                return
            texts = [d.text for d in batch]
            try:
                encodings = tokenizer.encode_batch(texts)
            except AttributeError:
                # Stub tokenizers (used in tests) may not implement encode_batch.
                # Fall back to single-doc encode — slow but correct.
                encodings = [tokenizer.encode(t) for t in texts]
            for doc, enc in zip(batch, encodings, strict=False):
                ids = enc.ids if hasattr(enc, "ids") else enc
                if not ids:
                    stats.docs_filtered += 1
                    continue
                stats.docs_kept += 1
                yield (doc.doc_id, source_id, ids,
                       _text_hash(doc.text), revision_id)

        for doc in _filtered_docs():
            _pending_batch.append(doc)
            if len(_pending_batch) >= tokenize_batch_size:
                yield from _flush(_pending_batch)
                _pending_batch.clear()
        # Final partial batch.
        if _pending_batch:
            yield from _flush(_pending_batch)
            _pending_batch.clear()

    for tokens, spans in pack_with_provenance(
        _kept_doc_stream(),
        sequence_length=sequence_length,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        drop_last=drop_last,
    ):
        writer.append_sequence(tokens, spans)
        stats.sequences_emitted += 1
        stats.tokens_emitted += sequence_length

    closed_shards = writer.close()
    # A per-source corpus is 100% one source by construction. The
    # production-mix target share lives in pretrain_mix.yaml + the
    # composed corpus's manifest, NOT here. (Earlier code stamped the
    # production share onto every per-source manifest, which made the
    # L5 source-share drift check fire spuriously on per-source
    # corpora — caught by 2026-05-12 smoke test.)
    # Pass the in-memory shard manifests so this works even when the
    # writer streamed-and-deleted local shards (the disk walk would
    # find nothing).
    manifest = write_corpus_manifest(
        output_dir,
        corpus_name=source_id,
        tokenizer_sha256=tokenizer_sha256,
        sequence_length=sequence_length,
        sequences_per_shard=sequences_per_shard,
        source_revisions={source_id: revision_id},
        target_source_share={source_id: 1.0},
        shard_manifests=closed_shards,
    )
    # If we streamed to R2, the top-level manifest.json also needs to
    # land in R2 (the per-shard hook only mirrors shard-* directories;
    # the top-level manifest is written AFTER the last shard close).
    if r2_prefix is not None:
        try:
            from myllm.utils.storage import upload_file
            upload_file(
                output_dir / "manifest.json",
                f"{r2_prefix.rstrip('/')}/manifest.json",
            )
            log.info("corpus_manifest_uploaded", prefix=r2_prefix)
        except Exception as e:  # noqa: BLE001
            log.error("corpus_manifest_upload_failed", error=str(e))
    # ``target_share`` (passed in) is informational — kept on the
    # BuildStats record so the compose pass / launcher can read it.
    stats.production_target_share = float(target_share)

    log.info(
        "build_one_source_done",
        source_id=source_id,
        docs_seen=stats.docs_seen,
        docs_kept=stats.docs_kept,
        docs_filtered=stats.docs_filtered,
        docs_contaminated=stats.docs_contaminated,
        docs_deduped=stats.docs_deduped,
        sequences_emitted=stats.sequences_emitted,
        tokens_emitted=stats.tokens_emitted,
    )
    return stats, manifest
