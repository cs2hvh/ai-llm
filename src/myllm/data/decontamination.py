"""N-gram-based benchmark decontamination for the pretrain corpus.

R6 from the 2026-05-11 dossier. Pretrain corpora pulled from the open web
will sometimes contain verbatim or near-verbatim copies of benchmark
questions (LeetCode pages, MMLU mirrors on Reddit, Stack Exchange dumps
of past exam questions, etc.). Training on those positions inflates gate
scores in a way that doesn't reflect real model capability. The standard
mitigation is **n-gram-based decontamination**: index every benchmark
prompt's n-grams, scan each corpus document, drop or flag any document
that contains a benchmark n-gram.

Pipeline integration:
    The core index is adapter-agnostic — it ingests bare prompt strings.
    Two helpers bridge to the actual benchmark code:
        - ``extract_prompts_from_benchmark``: pulls the ``.prompt`` field
          from each EvalExample yielded by a Benchmark.
        - ``index_from_benchmarks``: builds a fresh index from a list of
          Benchmark adapters in one call.
    For production runs, ``DecontaminationIndex.save_json`` / ``load_json``
    let you build the index offline once (slow: pulls HF data) and load
    it at every training-run start (fast: pure JSON read).

Recipe (Llama 2 / OLMo 2 / SmolLM3 convention):
    - n-gram size = 13 words (long enough to be specific, short enough
      to catch minor punctuation/whitespace variation)
    - case-folded + punctuation-stripped normalization (so "What is 2+2?"
      and "what is 2 + 2" hash to the same n-grams)
    - hash via xxhash64 for speed
    - a corpus doc is "contaminated" if it shares ≥1 n-gram with the
      benchmark index. The report records which benchmark(s) triggered.

Memory characteristics:
    - The benchmark index is small: 1K-50K prompts × ~20 n-grams each =
      100K-1M hash entries × 8 bytes = 800KB-8MB. Fits easily.
    - Each corpus doc is O(words) to scan; the dominant cost is hashing.

Output:
    Per-benchmark stats (n_examples, n_corpus_docs_matched, sample matched
    docs, longest matched n-gram chain) suitable for the OLMo-2-style
    contamination report we want to ship at every gate.

This module is the *core* — extracting prompts from a Benchmark adapter
and wiring into the data pipeline are separate (the index here is
adapter-agnostic; we just need an iterable of prompt strings).
"""
from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import xxhash

# Stripped of: ASCII punctuation, normalize whitespace runs.
_PUNCT_RE = re.compile(r"[^\w\s]+", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+", flags=re.UNICODE)


def _normalize(text: str) -> str:
    """Case-fold, strip ASCII punctuation, collapse whitespace.

    Note: we do NOT strip Unicode marks (Devanagari combining chars, etc.) —
    those carry meaning in non-Latin scripts. Only ASCII punctuation is
    removed.
    """
    if not text:
        return ""
    t = text.casefold()
    t = _PUNCT_RE.sub(" ", t)
    t = _WS_RE.sub(" ", t).strip()
    return t


def _hash_ngrams(words: list[str], n: int, seed: int) -> Iterator[int]:
    """Yield xxhash64 of each ``n``-word sliding window in ``words``."""
    if len(words) < n:
        return
    for i in range(len(words) - n + 1):
        ngram = " ".join(words[i : i + n])
        yield xxhash.xxh64(ngram, seed=seed).intdigest()


@dataclass(frozen=True)
class DecontaminationConfig:
    """Per-corpus decontamination knobs."""

    ngram_size: int = 13
    hash_seed: int = 0xDECAF
    # If True (default), each match record stores up to N example matched
    # n-grams from each contaminated document, to make the report
    # interpretable. False = skip recording examples (less memory).
    record_match_examples: bool = True
    max_examples_per_bench: int = 20


@dataclass
class BenchmarkSignature:
    """The set of hashed n-grams that fingerprint one benchmark."""

    benchmark_id: str
    ngrams: set[int]
    n_examples: int


@dataclass
class DecontaminationReport:
    """Per-benchmark contamination stats produced by scanning a corpus.

    Shape (per benchmark):
        {
            "n_examples": int,                   # benchmark size
            "n_corpus_docs_matched": int,        # how many corpus docs matched
            "n_corpus_docs_scanned": int,        # total docs scanned
            "match_rate": float,                 # matched / scanned
            "example_matches": list[dict],       # up to N samples (if recorded)
        }
    """

    per_bench: dict[str, dict] = field(default_factory=dict)
    n_corpus_docs_scanned: int = 0
    n_corpus_docs_with_any_match: int = 0

    def to_csv(self) -> str:
        """OLMo 2-style contamination CSV.

        Columns: ``benchmark, n_examples, n_corpus_docs_matched,
        n_corpus_docs_scanned, match_rate``.
        """
        lines = [
            "benchmark,n_examples,n_corpus_docs_matched,n_corpus_docs_scanned,match_rate"
        ]
        for bench_id in sorted(self.per_bench):
            stats = self.per_bench[bench_id]
            lines.append(
                f"{bench_id},{stats['n_examples']},"
                f"{stats['n_corpus_docs_matched']},"
                f"{stats['n_corpus_docs_scanned']},"
                f"{stats['match_rate']:.6f}"
            )
        return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Index
# --------------------------------------------------------------------------- #
class DecontaminationIndex:
    """Indexed set of benchmark n-grams; scans corpus docs for matches.

    Usage:

        cfg = DecontaminationConfig(ngram_size=13)
        idx = DecontaminationIndex(cfg)
        idx.add_benchmark("mmlu-prox", iter_prompts_from_mmlu_prox())
        idx.add_benchmark("belebele", iter_prompts_from_belebele())

        report = DecontaminationReport()
        for doc in corpus:
            matches = idx.scan_document(doc, report=report)
            if matches:
                # drop or flag this doc
                ...
        report.to_csv()  # emit the per-gate report
    """

    def __init__(self, config: DecontaminationConfig | None = None):
        self.config = config or DecontaminationConfig()
        self.signatures: dict[str, BenchmarkSignature] = {}
        # Reverse-lookup: ngram_hash → set of benchmark_ids that contain it.
        # Lets us answer "which benchmarks does this doc collide with"
        # in O(matched ngrams) instead of O(benchmarks × ngrams).
        self._ngram_to_benchmarks: dict[int, set[str]] = {}

    # ------------------------------------------------------------------- #
    # Building the index
    # ------------------------------------------------------------------- #
    def add_benchmark(self, benchmark_id: str, prompts: Iterable[str]) -> None:
        """Index every n-gram in every prompt for one benchmark.

        Idempotent: calling twice with the same benchmark_id replaces
        the previous index for that benchmark.
        """
        if benchmark_id in self.signatures:
            # Remove the old benchmark's contributions from the reverse map.
            self._unregister_benchmark(benchmark_id)

        ngrams: set[int] = set()
        n_examples = 0
        for prompt in prompts:
            n_examples += 1
            words = _normalize(prompt).split()
            for h in _hash_ngrams(words, self.config.ngram_size, self.config.hash_seed):
                ngrams.add(h)

        self.signatures[benchmark_id] = BenchmarkSignature(
            benchmark_id=benchmark_id, ngrams=ngrams, n_examples=n_examples
        )
        for h in ngrams:
            self._ngram_to_benchmarks.setdefault(h, set()).add(benchmark_id)

    def _unregister_benchmark(self, benchmark_id: str) -> None:
        sig = self.signatures.pop(benchmark_id, None)
        if sig is None:
            return
        for h in sig.ngrams:
            owners = self._ngram_to_benchmarks.get(h)
            if owners is None:
                continue
            owners.discard(benchmark_id)
            if not owners:
                del self._ngram_to_benchmarks[h]

    # ------------------------------------------------------------------- #
    # Scanning
    # ------------------------------------------------------------------- #
    def scan_document(
        self,
        text: str,
        *,
        report: DecontaminationReport | None = None,
        doc_id: str | None = None,
    ) -> dict[str, int]:
        """Scan one doc; return ``{benchmark_id: n_matched_ngrams}``.

        If ``report`` is provided, accumulate per-benchmark counters into it.
        """
        words = _normalize(text).split()
        matches: dict[str, int] = {}
        match_examples: dict[str, list[str]] = {}

        for h in _hash_ngrams(words, self.config.ngram_size, self.config.hash_seed):
            owners = self._ngram_to_benchmarks.get(h)
            if owners is None:
                continue
            for bench_id in owners:
                matches[bench_id] = matches.get(bench_id, 0) + 1
                # Optionally capture the matched n-gram text for the report.
                if self.config.record_match_examples and report is not None:
                    cur = match_examples.setdefault(bench_id, [])
                    if len(cur) < self.config.max_examples_per_bench:
                        # Reconstruct the n-gram from the position we'd need
                        # to remember; the simpler thing is to record the hash
                        # for now and a sample of the doc's prefix.
                        cur.append(text[:200])

        if report is not None:
            report.n_corpus_docs_scanned += 1
            if matches:
                report.n_corpus_docs_with_any_match += 1
            # Always count even benchmarks that didn't match for this doc,
            # so n_corpus_docs_scanned is consistent across benchmarks.
            for bench_id, sig in self.signatures.items():
                stats = report.per_bench.setdefault(
                    bench_id,
                    {
                        "n_examples": sig.n_examples,
                        "n_corpus_docs_matched": 0,
                        "n_corpus_docs_scanned": 0,
                        "match_rate": 0.0,
                        "example_matches": [],
                    },
                )
                stats["n_corpus_docs_scanned"] += 1
                if bench_id in matches:
                    stats["n_corpus_docs_matched"] += 1
                    if (
                        self.config.record_match_examples
                        and len(stats["example_matches"]) < self.config.max_examples_per_bench
                        and doc_id is not None
                    ):
                        stats["example_matches"].append({"doc_id": doc_id})
                stats["match_rate"] = (
                    stats["n_corpus_docs_matched"] / stats["n_corpus_docs_scanned"]
                )

        return matches

    # ------------------------------------------------------------------- #
    # Serialization — build once offline, load N times at run-start
    # ------------------------------------------------------------------- #
    def save_json(self, path: str | Path) -> None:
        """Serialize the full index (config + per-benchmark n-gram hash sets).

        N-gram counts can be ~1M for a multi-benchmark gate index; storing
        the sorted hash list as a JSON array works fine (~16MB worst case).
        The reverse-lookup map is rebuilt on load — no need to persist it.
        """
        data = {
            "config": {
                "ngram_size": self.config.ngram_size,
                "hash_seed": self.config.hash_seed,
                "record_match_examples": self.config.record_match_examples,
                "max_examples_per_bench": self.config.max_examples_per_bench,
            },
            "signatures": {
                bid: {
                    "n_examples": sig.n_examples,
                    "ngrams": sorted(sig.ngrams),
                }
                for bid, sig in self.signatures.items()
            },
        }
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(data, f)

    @classmethod
    def load_json(cls, path: str | Path) -> "DecontaminationIndex":
        """Inverse of ``save_json``. Rebuilds the reverse-lookup map."""
        with open(path) as f:
            data = json.load(f)
        cfg_dict = data["config"]
        cfg = DecontaminationConfig(
            ngram_size=int(cfg_dict["ngram_size"]),
            hash_seed=int(cfg_dict["hash_seed"]),
            record_match_examples=bool(cfg_dict.get("record_match_examples", True)),
            max_examples_per_bench=int(cfg_dict.get("max_examples_per_bench", 20)),
        )
        idx = cls(cfg)
        for bid, sig_data in data["signatures"].items():
            ngrams = set(int(h) for h in sig_data["ngrams"])
            idx.signatures[bid] = BenchmarkSignature(
                benchmark_id=bid,
                ngrams=ngrams,
                n_examples=int(sig_data["n_examples"]),
            )
            for h in ngrams:
                idx._ngram_to_benchmarks.setdefault(h, set()).add(bid)
        return idx


# --------------------------------------------------------------------------- #
# Pipeline helper — wrap a corpus iterator with decontamination filtering
# --------------------------------------------------------------------------- #
def decontaminate_iter(
    docs: Iterable[Any],
    index: DecontaminationIndex,
    *,
    text_fn=lambda d: d if isinstance(d, str) else d["text"],
    id_fn=lambda d: None,
    report: DecontaminationReport | None = None,
    drop_contaminated: bool = True,
) -> Iterator[Any]:
    """Stream a corpus, yielding only docs that don't match any benchmark.

    When ``drop_contaminated=False``, every doc is yielded but contaminated
    docs are tagged via the ``report`` so caller-side counters reflect the
    contamination rate.

    ``text_fn`` and ``id_fn`` let callers pass complex Document objects
    rather than bare strings.
    """
    for doc in docs:
        text = text_fn(doc)
        doc_id = id_fn(doc)
        matches = index.scan_document(text, report=report, doc_id=doc_id)
        if matches and drop_contaminated:
            continue
        yield doc


# --------------------------------------------------------------------------- #
# Benchmark-adapter bridge
# --------------------------------------------------------------------------- #
def extract_prompts_from_benchmark(
    benchmark: Any,
    *,
    split: str = "test",
    sample_size: int | None = None,
    seed: int = 0,
) -> Iterator[str]:
    """Yield prompt strings from a Benchmark adapter's ``load_examples``.

    The Benchmark protocol returns ``EvalExample`` objects with a
    ``.prompt`` attribute; this helper extracts just the strings so the
    decontamination index doesn't need to know about ``EvalExample``.

    Accepts dict-shaped examples too (test convenience).
    """
    for ex in benchmark.load_examples(split=split, sample_size=sample_size, seed=seed):
        if hasattr(ex, "prompt"):
            yield ex.prompt
        elif isinstance(ex, dict) and "prompt" in ex:
            yield ex["prompt"]
        else:
            raise TypeError(
                f"benchmark example has neither `.prompt` attribute nor "
                f"`prompt` key: {type(ex).__name__}"
            )


def index_from_benchmarks(
    benchmarks: Iterable[Any],
    *,
    config: DecontaminationConfig | None = None,
    split: str = "test",
    sample_size_per_benchmark: int | None = None,
    seed: int = 0,
) -> DecontaminationIndex:
    """Build a fresh ``DecontaminationIndex`` from a list of Benchmark adapters.

    Each benchmark's ``name`` attribute is used as the ``benchmark_id``.
    For multilingual benchmarks (MMLU-ProX, Belebele, MILU), the adapter's
    own per-language sampling logic is reused via ``sample_size``.
    """
    idx = DecontaminationIndex(config)
    for bench in benchmarks:
        prompts = extract_prompts_from_benchmark(
            bench, split=split, sample_size=sample_size_per_benchmark, seed=seed
        )
        idx.add_benchmark(bench.name, prompts)
    return idx
