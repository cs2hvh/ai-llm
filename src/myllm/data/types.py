"""Shared dataclasses for the data pipeline.

These are the *only* types that cross module boundaries inside ``myllm.data``.
Module-internal data structures stay private to their module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

DocumentSource = Literal[
    "web",
    "code",
    "wiki",
    "books",
    "academic",
    "math",
    "qa",
    "multilingual",
    "structured",
]


@dataclass
class Document:
    """A single document flowing through the pipeline.

    ``text`` is mutable — filters that redact (e.g. PII) update it in place.
    Filters that only judge keep/reject must not modify ``text``.
    """

    text: str
    source: DocumentSource
    dataset: str
    doc_id: str
    language: str | None = None
    metadata: dict[str, str | int | float] = field(default_factory=dict)


@dataclass(frozen=True)
class FilterDecision:
    keep: bool
    reason: str
    score: float | None = None


@dataclass(frozen=True)
class ShardSpec:
    """A processable unit of source data."""

    shard_id: str
    dataset: str
    category: DocumentSource
    split: str = "train"
    files: tuple[str, ...] = ()
    estimated_bytes: int | None = None


@dataclass(frozen=True)
class ProcessedShardManifest:
    """Persistent record of a processed shard, written atomically on completion."""

    shard_id: str
    dataset: str
    input_doc_count: int
    kept_doc_count: int
    rejected_doc_count: int
    rejected_by_reason: dict[str, int]
    output_path: str
    output_sha256: str
    processed_at_iso: str
    pipeline_version: str
