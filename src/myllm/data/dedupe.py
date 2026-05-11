"""MinHash + LSH near-duplicate detection.

Algorithm (standard MinHash-LSH):
    1. Tokenize each document into a set of overlapping ``ngram_size``-word
       shingles.
    2. Compute ``num_perm`` MinHash signatures per document by hashing each
       shingle ``num_perm`` times with independent permutations and keeping
       the per-permutation minimum.
    3. Bucket signatures into ``num_bands`` bands of ``rows_per_band`` rows.
       Two documents whose signatures match exactly in any band are
       candidates; verify Jaccard via the full signature.
    4. Documents with a verified Jaccard ≥ ``threshold`` are duplicates of
       a previously-seen document.

References: Broder 1997, "On the resemblance and containment of documents";
Leskovec, Rajaraman, Ullman, "Mining of Massive Datasets" ch. 3.

Notes:
    - We use ``xxhash.xxh64`` for fast non-cryptographic hashing. Permutation
      parameters are derived from a master seed so a corpus can be re-deduped
      reproducibly.
    - The in-memory index scales linearly with documents. For corpora past a
      few hundred million docs, swap the in-memory dicts for an external
      key-value store (RocksDB, Redis) — same algorithm, different backend.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator

import xxhash

from myllm.utils.exceptions import DedupeError


def _shingles(text: str, ngram_size: int) -> set[str]:
    """Return the set of ``ngram_size``-word shingles from ``text``."""
    words = text.split()
    if len(words) < ngram_size:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + ngram_size]) for i in range(len(words) - ngram_size + 1)}


@dataclass(frozen=True)
class MinHashConfig:
    num_perm: int = 128
    num_bands: int = 32
    ngram_size: int = 5
    threshold: float = 0.85
    seed: int = 0xC0FFEE

    def __post_init__(self) -> None:
        if self.num_perm % self.num_bands != 0:
            raise ValueError(
                f"num_perm ({self.num_perm}) must be divisible by num_bands "
                f"({self.num_bands})"
            )
        if not 0 < self.threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        if self.ngram_size < 1:
            raise ValueError("ngram_size must be >= 1")

    @property
    def rows_per_band(self) -> int:
        return self.num_perm // self.num_bands


@dataclass
class MinHasher:
    """Compute MinHash signatures using ``num_perm`` independent hash seeds."""

    config: MinHashConfig
    _seeds: tuple[int, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        rng = xxhash.xxh64(self.config.seed.to_bytes(8, "little"))
        self._seeds = tuple(
            int.from_bytes(
                xxhash.xxh64(
                    rng.intdigest().to_bytes(8, "little") + i.to_bytes(8, "little"),
                ).digest(),
                "little",
            )
            for i in range(self.config.num_perm)
        )

    def signature(self, text: str) -> tuple[int, ...]:
        shingles = _shingles(text, self.config.ngram_size)
        if not shingles:
            return tuple(0 for _ in range(self.config.num_perm))
        seeds = self._seeds
        signature = [(1 << 64) - 1] * self.config.num_perm
        for shingle in shingles:
            sb = shingle.encode("utf-8")
            for i, seed in enumerate(seeds):
                h = xxhash.xxh64(sb, seed=seed).intdigest()
                if h < signature[i]:
                    signature[i] = h
        return tuple(signature)


def jaccard_from_signatures(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    """Estimate Jaccard similarity from two MinHash signatures of equal length."""
    if len(a) != len(b):
        raise DedupeError(f"signature length mismatch: {len(a)} vs {len(b)}")
    if not a:
        return 0.0
    matches = sum(1 for x, y in zip(a, b) if x == y)
    return matches / len(a)


class LSHIndex:
    """In-memory locality-sensitive hash index over MinHash signatures."""

    def __init__(self, config: MinHashConfig) -> None:
        self.config = config
        # bands[band_idx] = { band_signature_tuple: list of doc_ids }
        self._bands: list[dict[tuple[int, ...], list[str]]] = [
            {} for _ in range(config.num_bands)
        ]
        self._signatures: dict[str, tuple[int, ...]] = {}

    def __len__(self) -> int:
        return len(self._signatures)

    def add(self, doc_id: str, signature: tuple[int, ...]) -> None:
        if doc_id in self._signatures:
            raise DedupeError(f"doc_id already indexed: {doc_id}")
        if len(signature) != self.config.num_perm:
            raise DedupeError(
                f"signature length {len(signature)} != num_perm {self.config.num_perm}"
            )
        self._signatures[doc_id] = signature
        for band_idx, band_sig in enumerate(self._iter_bands(signature)):
            self._bands[band_idx].setdefault(band_sig, []).append(doc_id)

    def candidate_duplicates(self, signature: tuple[int, ...]) -> set[str]:
        candidates: set[str] = set()
        for band_idx, band_sig in enumerate(self._iter_bands(signature)):
            bucket = self._bands[band_idx].get(band_sig)
            if bucket:
                candidates.update(bucket)
        return candidates

    def find_duplicate(self, signature: tuple[int, ...]) -> str | None:
        """Return the first existing doc_id whose Jaccard ≥ threshold, else None."""
        for cand in self.candidate_duplicates(signature):
            if (
                jaccard_from_signatures(signature, self._signatures[cand])
                >= self.config.threshold
            ):
                return cand
        return None

    def _iter_bands(self, signature: tuple[int, ...]) -> Iterator[tuple[int, ...]]:
        rows = self.config.rows_per_band
        for band_idx in range(self.config.num_bands):
            yield signature[band_idx * rows : (band_idx + 1) * rows]


class Deduplicator:
    """High-level dedupe API. ``add_if_new`` returns False if the doc is a dup."""

    def __init__(self, config: MinHashConfig | None = None) -> None:
        self.config = config or MinHashConfig()
        self._hasher = MinHasher(self.config)
        self._index = LSHIndex(self.config)

    def add_if_new(self, doc_id: str, text: str) -> tuple[bool, str | None]:
        """Add document if not a duplicate. Returns ``(added, dup_of_doc_id)``."""
        sig = self._hasher.signature(text)
        existing = self._index.find_duplicate(sig)
        if existing is not None:
            return False, existing
        self._index.add(doc_id, sig)
        return True, None

    def filter_iter(
        self, docs: Iterable[tuple[str, str]]
    ) -> Iterator[tuple[str, str, bool, str | None]]:
        """Stream over (doc_id, text); yield (doc_id, text, kept, dup_of)."""
        for doc_id, text in docs:
            added, dup_of = self.add_if_new(doc_id, text)
            yield doc_id, text, added, dup_of

    def __len__(self) -> int:
        return len(self._index)
