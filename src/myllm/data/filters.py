"""Document filters — abstract base + concrete heuristic implementations.

A filter takes a ``Document`` and returns a ``FilterDecision``. Filters MUST
be deterministic given their config. ``PIIRedactor`` is the only filter
that mutates ``doc.text`` (intentional — redaction preserves the document
for training).

Compose filters via ``FilterChain``, which short-circuits on first reject
and surfaces the rejection reason for telemetry.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

from myllm.data.types import Document, FilterDecision

if TYPE_CHECKING:
    from myllm.data.decontamination import (
        DecontaminationIndex,
        DecontaminationReport,
    )

PIPELINE_VERSION = "filters/v0.1"


class Filter(ABC):
    """A document-level filter."""

    name: str = "filter"

    @abstractmethod
    def apply(self, doc: Document) -> FilterDecision:
        ...


# --------------------------------------------------------------------------- #
# Length
# --------------------------------------------------------------------------- #
@dataclass
class LengthFilter(Filter):
    """Reject documents outside [min_chars, max_chars]."""

    min_chars: int = 200
    max_chars: int = 1_000_000
    name: str = "length"

    def __post_init__(self) -> None:
        if self.min_chars < 0 or self.max_chars <= self.min_chars:
            raise ValueError(
                f"invalid length bounds: min={self.min_chars}, max={self.max_chars}"
            )

    def apply(self, doc: Document) -> FilterDecision:
        n = len(doc.text)
        if n < self.min_chars:
            return FilterDecision(False, "too_short", float(n))
        if n > self.max_chars:
            return FilterDecision(False, "too_long", float(n))
        return FilterDecision(True, "ok", float(n))


# --------------------------------------------------------------------------- #
# Repetition
# --------------------------------------------------------------------------- #
@dataclass
class RepetitionFilter(Filter):
    """Reject documents whose top word or top n-gram dominates the content."""

    max_top_word_share: float = 0.20
    max_top_ngram_share: float = 0.10
    ngram_n: int = 5
    name: str = "repetition"

    def __post_init__(self) -> None:
        if not 0 < self.max_top_word_share <= 1 or not 0 < self.max_top_ngram_share <= 1:
            raise ValueError("share thresholds must be in (0, 1]")
        if self.ngram_n < 2:
            raise ValueError("ngram_n must be >= 2")

    def apply(self, doc: Document) -> FilterDecision:
        words = doc.text.split()
        if not words:
            return FilterDecision(False, "empty", 0.0)
        wc = Counter(words)
        top_word_share = wc.most_common(1)[0][1] / len(words)
        if top_word_share > self.max_top_word_share:
            return FilterDecision(False, "top_word_repeated", top_word_share)
        if len(words) >= self.ngram_n:
            ng = Counter(
                tuple(words[i : i + self.ngram_n])
                for i in range(len(words) - self.ngram_n + 1)
            )
            total_ngrams = len(words) - self.ngram_n + 1
            top_ng_share = ng.most_common(1)[0][1] / total_ngrams
            if top_ng_share > self.max_top_ngram_share:
                return FilterDecision(False, "top_ngram_repeated", top_ng_share)
        return FilterDecision(True, "ok", top_word_share)


# --------------------------------------------------------------------------- #
# Symbol ratio
# --------------------------------------------------------------------------- #
@dataclass
class SymbolRatioFilter(Filter):
    """Reject documents with too many non-alphanumeric, non-whitespace characters."""

    max_symbol_ratio: float = 0.30
    name: str = "symbol_ratio"

    def __post_init__(self) -> None:
        if not 0 < self.max_symbol_ratio < 1:
            raise ValueError("max_symbol_ratio must be in (0, 1)")

    def apply(self, doc: Document) -> FilterDecision:
        if not doc.text:
            return FilterDecision(False, "empty", 0.0)
        n_symbol = sum(1 for c in doc.text if not c.isalnum() and not c.isspace())
        ratio = n_symbol / len(doc.text)
        if ratio > self.max_symbol_ratio:
            return FilterDecision(False, "too_symbolic", ratio)
        return FilterDecision(True, "ok", ratio)


# --------------------------------------------------------------------------- #
# PII redaction
# --------------------------------------------------------------------------- #
_PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "phone": re.compile(
        r"\b(?:\+?\d{1,3}[\s-]?)?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{4}\b"
    ),
    "ipv4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}


@dataclass
class PIIRedactor(Filter):
    """Redact common PII patterns. Mutates ``doc.text`` in place; never rejects.

    For high-stakes fields (SSN, credit-card, government IDs) use a real PII
    library such as Microsoft Presidio. This is a baseline web-scale filter.
    """

    redact_email: bool = True
    redact_phone: bool = True
    redact_ipv4: bool = False
    name: str = "pii_redact"

    def apply(self, doc: Document) -> FilterDecision:
        text = doc.text
        n_redactions = 0
        if self.redact_email:
            text, k = _PII_PATTERNS["email"].subn("<EMAIL>", text)
            n_redactions += k
        if self.redact_phone:
            text, k = _PII_PATTERNS["phone"].subn("<PHONE>", text)
            n_redactions += k
        if self.redact_ipv4:
            text, k = _PII_PATTERNS["ipv4"].subn("<IPV4>", text)
            n_redactions += k
        doc.text = text
        return FilterDecision(True, "ok", float(n_redactions))


# --------------------------------------------------------------------------- #
# Benchmark-prompt decontamination (R6)
# --------------------------------------------------------------------------- #
@dataclass
class DecontaminationFilter(Filter):
    """Reject documents that share an n-gram with any indexed benchmark.

    Wraps a pre-built ``DecontaminationIndex``. On each apply, scans the
    document's text and, if a benchmark n-gram matches, returns a reject
    decision with ``reason="contaminated"`` and ``score=n_total_matches``.

    The per-benchmark attribution lives in the optional ``report``; pass
    a shared ``DecontaminationReport`` instance to accumulate stats across
    the whole stream for the gate CSV.

    Position in the chain: place after the cheap structural filters
    (length, repetition, symbol-ratio) but before ``PIIRedactor`` so the
    PII redactor doesn't get called on docs we'd reject anyway.
    """

    index: "DecontaminationIndex"
    report: "DecontaminationReport | None" = None
    name: str = "decontamination"

    def apply(self, doc: Document) -> FilterDecision:
        matches = self.index.scan_document(
            doc.text,
            report=self.report,
            doc_id=doc.doc_id,
        )
        if matches:
            n_total = float(sum(matches.values()))
            return FilterDecision(False, "contaminated", n_total)
        return FilterDecision(True, "ok", 0.0)


# --------------------------------------------------------------------------- #
# Chain
# --------------------------------------------------------------------------- #
@dataclass
class FilterChain:
    """Apply filters in order; short-circuit on first reject.

    Returns the (possibly mutated) document together with the final decision.
    """

    filters: tuple[Filter, ...]

    def __post_init__(self) -> None:
        if not self.filters:
            raise ValueError("FilterChain must contain at least one filter")

    def apply(self, doc: Document) -> tuple[Document, FilterDecision]:
        for f in self.filters:
            decision = f.apply(doc)
            if not decision.keep:
                return doc, decision
        return doc, FilterDecision(True, "ok")
