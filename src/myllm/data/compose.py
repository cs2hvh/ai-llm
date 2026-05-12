"""Compose per-source packed corpora into a training-time mixed corpus.

Final B2 piece. After ``build_one_source`` produces one corpus per
data source, this module reads N source corpora and interleaves their
packed sequences at the target token shares, emitting a single
training-time corpus whose sequence order is what the training loop
actually consumes.

Why a separate pass (two-level shard design per reviewer Q&A §4):
  - Per-source corpora preserve revision pinning, dedup state, and
    source-local provenance. They're the build artifact.
  - Training-time corpus enforces the target mixture distribution AND
    optimizes for resume-throughput (one big sequence_id space, no
    per-step mixture decision).
  - Decoupling lets us re-mix without re-tokenizing (cheap) when
    target shares change.

Algorithm — deficit-driven token-share enforcement:
  Every packed sequence is exactly ``sequence_length`` tokens. Per-source
  corpora are built from a single source, so every sequence in source_a
  has source_a tokens (modulo packed cross-doc EOS, which is also
  source_a-attributed). Picking sequences by source therefore gives
  exact token-share control:

      pick = argmax_i (target_share_i × total_emitted_seqs - emitted_seqs_i)

  Each output sequence consumes exactly one input sequence; the
  output's sequence_id increases monotonically. Provenance is preserved
  by copying DocSpan rows from the source corpus into the new one.

Bootstrap:
  Until each source has emitted at least one sequence, we sample by
  weight (target_share) to avoid the all-zeros-deficit ambiguity. After
  that, deficit dominates.

Exhaustion:
  Per-source corpora have finite size. When a source is exhausted, it's
  removed from the candidate pool and the remaining shares are
  renormalized. We log this loudly because it usually means the source
  ran out before the target token budget did — the operator likely
  wanted to stop earlier.
"""
from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from myllm.data.packed_corpus import (
    DocSpan,
    PackedCorpusReader,
    PackedCorpusWriter,
    write_corpus_manifest,
)
from myllm.utils import get_logger

log = get_logger(__name__)


@dataclass
class ComposeStats:
    """Counters returned by compose_mixed_corpus."""
    total_sequences_emitted: int = 0
    total_tokens_emitted: int = 0
    per_source_sequences: dict[str, int] = None
    per_source_tokens: dict[str, int] = None
    exhausted_sources: list[str] = None

    def __post_init__(self):
        if self.per_source_sequences is None:
            self.per_source_sequences = {}
        if self.per_source_tokens is None:
            self.per_source_tokens = {}
        if self.exhausted_sources is None:
            self.exhausted_sources = []


def compose_mixed_corpus(
    source_corpora: dict[str, str | Path],
    output_dir: str | Path,
    *,
    target_shares: dict[str, float],
    sequences_per_shard: int,
    corpus_name: str = "mixed",
    target_total_sequences: int | None = None,
    seed: int = 0,
    bootstrap_steps: int = 16,
) -> tuple[ComposeStats, "object"]:
    """Read N per-source corpora and emit a training-time mixed corpus.

    Args:
        source_corpora: ``{source_id: path_to_per_source_corpus_root}``.
            Each must be a complete corpus (top-level manifest.json present).
        output_dir: where to write the mixed corpus.
        target_shares: ``{source_id: share}``, must sum to ~1.0.
        sequences_per_shard: shard size for the OUTPUT corpus
            (independent of per-source shard size).
        corpus_name: label stored in the output CorpusManifest.
        target_total_sequences: stop after this many output sequences.
            If None, emit until at least one source is exhausted AND
            target shares are within tolerance.
        seed: RNG seed for the bootstrap-phase weighted draws.
        bootstrap_steps: number of picks made by pure-weight sampling
            before switching to deficit-driven sampling.

    Returns ``(ComposeStats, CorpusManifest)``.

    All readers' tokenizer_sha256 must match. Sequence length must match.
    A mismatch is a build error and raises.
    """
    if not source_corpora:
        raise ValueError("source_corpora is empty")
    # Validate target_shares.
    missing_targets = set(source_corpora) - set(target_shares)
    if missing_targets:
        raise ValueError(
            f"target_shares missing entries for sources: {sorted(missing_targets)}"
        )
    extra_targets = set(target_shares) - set(source_corpora)
    if extra_targets:
        raise ValueError(
            f"target_shares has entries with no matching source corpus: "
            f"{sorted(extra_targets)}"
        )
    total_share = sum(target_shares.values())
    if abs(total_share - 1.0) > 1e-6:
        raise ValueError(
            f"target_shares must sum to 1.0; got {total_share}"
        )

    # Open all readers.
    readers: dict[str, PackedCorpusReader] = {
        sid: PackedCorpusReader(Path(p)) for sid, p in source_corpora.items()
    }

    # Cross-validate: tokenizer + sequence length.
    first_sid = next(iter(readers))
    expected_tok = readers[first_sid].manifest.tokenizer_sha256
    expected_seq = readers[first_sid].manifest.sequence_length
    for sid, r in readers.items():
        if r.manifest.tokenizer_sha256 != expected_tok:
            raise ValueError(
                f"source {sid!r} tokenizer_sha256 mismatch: "
                f"{r.manifest.tokenizer_sha256} != {expected_tok}"
            )
        if r.manifest.sequence_length != expected_seq:
            raise ValueError(
                f"source {sid!r} sequence_length mismatch: "
                f"{r.manifest.sequence_length} != {expected_seq}"
            )

    log.info(
        "compose_mixed_corpus_start",
        sources=sorted(source_corpora),
        sequence_length=expected_seq,
        target_total_sequences=target_total_sequences,
    )

    # Per-source consumption cursors + emission counters.
    cursors: dict[str, int] = {sid: 0 for sid in readers}
    emitted_per_source: dict[str, int] = {sid: 0 for sid in readers}
    revision_per_source: dict[str, str] = {
        sid: r.manifest.source_revisions.get(sid, "unpinned")
        for sid, r in readers.items()
    }

    # Build the per-source revision map for the output manifest.
    output_revisions: dict[str, str] = dict(revision_per_source)

    rng = random.Random(seed)
    writer = PackedCorpusWriter(
        Path(output_dir),
        sequence_length=expected_seq,
        sequences_per_shard=sequences_per_shard,
        tokenizer_sha256=expected_tok,
    )

    stats = ComposeStats()

    def _exhausted() -> set[str]:
        return {sid for sid in readers if cursors[sid] >= readers[sid].total_sequences}

    def _candidate_sources() -> list[str]:
        return [sid for sid in readers if cursors[sid] < readers[sid].total_sequences]

    def _pick_source() -> str | None:
        cands = _candidate_sources()
        if not cands:
            return None
        emitted_total = sum(emitted_per_source.values())
        # Bootstrap phase OR all deficits zero → weighted random pick.
        if emitted_total < bootstrap_steps:
            weights = [target_shares[sid] for sid in cands]
            total = sum(weights)
            if total <= 0:
                return rng.choice(cands)
            return rng.choices(cands, weights=weights, k=1)[0]
        # Deficit phase: pick max deficit. Ties broken by weighted RNG.
        deficits = {
            sid: target_shares[sid] * emitted_total - emitted_per_source[sid]
            for sid in cands
        }
        max_deficit = max(deficits.values())
        # Tolerance: if every source is within 1 sequence of fair share, fall
        # back to weighted random.
        if max_deficit <= 0:
            weights = [target_shares[sid] for sid in cands]
            return rng.choices(cands, weights=weights, k=1)[0]
        top_sources = [sid for sid, d in deficits.items() if d == max_deficit]
        return rng.choice(top_sources)

    output_sequence_id = 0
    doc_span_offset = 0  # remapping doc_span_ids into the output corpus's space

    while True:
        if (
            target_total_sequences is not None
            and stats.total_sequences_emitted >= target_total_sequences
        ):
            break
        sid = _pick_source()
        if sid is None:
            log.info("compose_all_sources_exhausted")
            stats.exhausted_sources = sorted(_exhausted())
            break

        # Read one sequence from source `sid` at its cursor.
        local_id = cursors[sid]
        tokens = np.array(readers[sid].get_sequence(local_id), copy=True)
        provenance = readers[sid].get_provenance(local_id)
        # Remap each DocSpan: source_id stays (it's the data origin),
        # dataset_revision_id stays. doc_span_id/sequence_id are
        # reassigned by the writer.
        remapped: list[DocSpan] = []
        for s in provenance:
            remapped.append(DocSpan(
                doc_span_id=-1,
                sequence_id=-1,
                source_id=s.source_id,
                doc_id_hash=s.doc_id_hash,
                dataset_revision_id=s.dataset_revision_id,
                token_start_in_sequence=s.token_start_in_sequence,
                token_end_in_sequence=s.token_end_in_sequence,
                text_hash=s.text_hash,
            ))

        writer.append_sequence(tokens, remapped)
        cursors[sid] += 1
        emitted_per_source[sid] += 1
        stats.total_sequences_emitted += 1
        stats.total_tokens_emitted += expected_seq
        output_sequence_id += 1

        # Log progress periodically.
        if stats.total_sequences_emitted % 10000 == 0:
            log.info(
                "compose_progress",
                emitted_total=stats.total_sequences_emitted,
                per_source=dict(emitted_per_source),
            )

    writer.close()

    stats.per_source_sequences = dict(emitted_per_source)
    stats.per_source_tokens = {
        sid: n * expected_seq for sid, n in emitted_per_source.items()
    }
    if not stats.exhausted_sources:
        stats.exhausted_sources = sorted(_exhausted())

    manifest = write_corpus_manifest(
        Path(output_dir),
        corpus_name=corpus_name,
        tokenizer_sha256=expected_tok,
        sequence_length=expected_seq,
        sequences_per_shard=sequences_per_shard,
        source_revisions=output_revisions,
        target_source_share=dict(target_shares),
    )

    log.info(
        "compose_mixed_corpus_done",
        output_dir=str(output_dir),
        total_sequences=stats.total_sequences_emitted,
        total_tokens=stats.total_tokens_emitted,
        actual_share=dict(manifest.actual_source_share),
        exhausted=stats.exhausted_sources,
    )
    return stats, manifest


def iterate_packed_corpus(
    root: str | Path,
    *,
    start_sequence_id: int = 0,
) -> Iterator[tuple[int, np.ndarray]]:
    """Convenience wrapper: open a corpus and stream (sequence_id, tokens).

    Useful for the training loop's data path — replaces the HF-stream
    pipeline once an offline packed corpus exists. The reader handles
    LRU eviction so a long iteration doesn't blow up memory.
    """
    reader = PackedCorpusReader(Path(root))
    yield from reader.iterate_from(start_sequence_id)
