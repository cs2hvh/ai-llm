"""Regression tests for benchmark n-gram decontamination (R6).

Validates:
  - Normalization (case + punctuation invariance).
  - Index build + scan: docs that share an n-gram with a benchmark match;
    clean docs don't.
  - Multi-benchmark attribution: when a doc's n-gram appears in two
    benchmarks, both get credited.
  - Per-benchmark report aggregation: n_examples, matched, scanned, rate.
  - CSV report output format (OLMo-2 style).
  - decontaminate_iter wraps a corpus, drops contaminated docs by default,
    can be flipped to tag-only.
"""
from __future__ import annotations

import pytest

from myllm.data.decontamination import (
    DecontaminationConfig,
    DecontaminationIndex,
    DecontaminationReport,
    decontaminate_iter,
)


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #
def test_case_invariant_match():
    """A benchmark prompt in title case should match a corpus doc in lowercase."""
    idx = DecontaminationIndex(DecontaminationConfig(ngram_size=4))
    idx.add_benchmark("bench", ["What Is The Capital Of France"])
    matches = idx.scan_document("Today's lesson: what is the capital of france is paris.")
    assert "bench" in matches


def test_punctuation_invariant_match():
    """Punctuation differences shouldn't break a match."""
    idx = DecontaminationIndex(DecontaminationConfig(ngram_size=4))
    idx.add_benchmark("bench", ["what is the capital of France?"])
    matches = idx.scan_document("Question: what is the capital of france??? Answer: Paris.")
    assert "bench" in matches


def test_unicode_words_preserved():
    """Devanagari and other non-Latin text should still tokenize and match."""
    idx = DecontaminationIndex(DecontaminationConfig(ngram_size=3))
    idx.add_benchmark("hi-bench", ["भारत की राजधानी क्या है"])
    matches = idx.scan_document("निम्न प्रश्न का उत्तर दें: भारत की राजधानी क्या है? उत्तर: दिल्ली।")
    assert "hi-bench" in matches


# --------------------------------------------------------------------------- #
# Basic scan correctness
# --------------------------------------------------------------------------- #
def test_clean_doc_returns_empty_dict():
    idx = DecontaminationIndex(DecontaminationConfig(ngram_size=5))
    idx.add_benchmark("mmlu", ["The mitochondria is the powerhouse of the cell."])
    matches = idx.scan_document("Quick brown fox jumps over the lazy dog. Pack my box.")
    assert matches == {}


def test_short_doc_no_match():
    """A doc shorter than n_words can never match (no n-grams to compare)."""
    idx = DecontaminationIndex(DecontaminationConfig(ngram_size=10))
    idx.add_benchmark("bench", ["this is a benchmark question with more than ten words in it"])
    matches = idx.scan_document("five word doc only here")
    assert matches == {}


def test_matched_doc_returns_count_per_benchmark():
    """Multi-ngram match: a 20-word benchmark embedded in a doc should hit
    multiple sliding-window n-grams."""
    idx = DecontaminationIndex(DecontaminationConfig(ngram_size=5))
    idx.add_benchmark("bench", ["the quick brown fox jumps over the lazy dog by the river"])
    # Embed the entire benchmark phrase verbatim.
    doc = "Earlier in the day the quick brown fox jumps over the lazy dog by the river and disappears."
    matches = idx.scan_document(doc)
    assert matches.get("bench", 0) >= 5  # at least 5 sliding 5-grams should hit


# --------------------------------------------------------------------------- #
# Multi-benchmark attribution
# --------------------------------------------------------------------------- #
def test_doc_matching_multiple_benchmarks_credited_to_each():
    """A doc containing prompts from two benchmarks should match BOTH."""
    idx = DecontaminationIndex(DecontaminationConfig(ngram_size=4))
    idx.add_benchmark("bench-A", ["alpha beta gamma delta epsilon zeta"])
    idx.add_benchmark("bench-B", ["one two three four five six"])
    doc = "Text: alpha beta gamma delta epsilon zeta and also one two three four five six end."
    matches = idx.scan_document(doc)
    assert "bench-A" in matches
    assert "bench-B" in matches


def test_shared_ngram_credited_to_both_benchmarks():
    """If two benchmarks happen to share an exact n-gram (unlikely but
    possible), a doc with that n-gram should be credited to both."""
    shared = "first sentence with shared content shared content shared content"
    idx = DecontaminationIndex(DecontaminationConfig(ngram_size=3))
    idx.add_benchmark("bench-A", [shared])
    idx.add_benchmark("bench-B", [shared])
    matches = idx.scan_document(shared)
    assert "bench-A" in matches
    assert "bench-B" in matches


# --------------------------------------------------------------------------- #
# Index management
# --------------------------------------------------------------------------- #
def test_add_benchmark_twice_replaces():
    """Re-adding a benchmark with new prompts replaces the old ones."""
    idx = DecontaminationIndex(DecontaminationConfig(ngram_size=4))
    idx.add_benchmark("bench", ["first prompt here that contains many words for matching"])
    assert "first prompt here that contains many words for matching" in (
        " ".join(p for p in ["first prompt here that contains many words for matching"])
    )
    # Replace with a different prompt.
    idx.add_benchmark("bench", ["different prompt entirely with no words from the old one at all"])
    # Old prompt no longer matches.
    matches = idx.scan_document("Hello first prompt here that contains many words for matching end.")
    assert matches.get("bench", 0) == 0
    # New prompt now matches.
    matches = idx.scan_document("Hello different prompt entirely with no words from the old one at all end.")
    assert matches.get("bench", 0) >= 1


# --------------------------------------------------------------------------- #
# Report aggregation
# --------------------------------------------------------------------------- #
def test_report_aggregates_per_benchmark():
    idx = DecontaminationIndex(DecontaminationConfig(ngram_size=4))
    idx.add_benchmark("mmlu", ["what is the capital of france"])
    idx.add_benchmark("gsm", ["how many apples does john have left"])
    report = DecontaminationReport()

    # Scan 10 docs; 3 match mmlu, 1 matches gsm, 1 matches both.
    docs = [
        "what is the capital of france and germany",       # mmlu
        "what is the capital of france and italy",         # mmlu
        "what is the capital of france and spain",         # mmlu
        "how many apples does john have left after giving", # gsm
        "what is the capital of france and how many apples does john have left", # both
        "completely unrelated content number one",         # clean
        "completely unrelated content number two",         # clean
        "completely unrelated content number three",       # clean
        "completely unrelated content number four",        # clean
        "completely unrelated content number five",        # clean
    ]
    for doc in docs:
        idx.scan_document(doc, report=report)

    assert report.n_corpus_docs_scanned == 10
    assert report.n_corpus_docs_with_any_match == 5
    assert report.per_bench["mmlu"]["n_corpus_docs_matched"] == 4
    assert report.per_bench["gsm"]["n_corpus_docs_matched"] == 2
    # Match rates.
    assert report.per_bench["mmlu"]["match_rate"] == pytest.approx(0.4)
    assert report.per_bench["gsm"]["match_rate"] == pytest.approx(0.2)


def test_report_csv_format():
    idx = DecontaminationIndex(DecontaminationConfig(ngram_size=4))
    idx.add_benchmark("bench-a", ["alpha beta gamma delta epsilon"])
    idx.add_benchmark("bench-b", ["one two three four five"])
    report = DecontaminationReport()
    idx.scan_document("alpha beta gamma delta epsilon end", report=report)
    idx.scan_document("clean doc no matches here", report=report)

    csv = report.to_csv()
    lines = csv.strip().split("\n")
    assert lines[0] == "benchmark,n_examples,n_corpus_docs_matched,n_corpus_docs_scanned,match_rate"
    # 2 benchmarks, sorted alphabetically.
    assert lines[1].startswith("bench-a,")
    assert lines[2].startswith("bench-b,")
    # Each row has 5 fields.
    for line in lines[1:]:
        assert len(line.split(",")) == 5


# --------------------------------------------------------------------------- #
# decontaminate_iter: pipeline wrapper
# --------------------------------------------------------------------------- #
def test_decontaminate_iter_drops_matched_docs_by_default():
    idx = DecontaminationIndex(DecontaminationConfig(ngram_size=4))
    idx.add_benchmark("bench", ["the capital of france is paris"])
    docs = [
        "clean doc one",
        "another sentence with the capital of france is paris inside",   # matches
        "clean doc two",
        "yet another match: the capital of france is paris guaranteed",  # matches
        "clean doc three",
    ]
    kept = list(decontaminate_iter(docs, idx))
    assert kept == ["clean doc one", "clean doc two", "clean doc three"]


def test_decontaminate_iter_can_tag_only():
    """drop_contaminated=False yields ALL docs, but populates the report."""
    idx = DecontaminationIndex(DecontaminationConfig(ngram_size=4))
    idx.add_benchmark("bench", ["the capital of france is paris"])
    docs = [
        "clean doc one",
        "another sentence with the capital of france is paris inside",
        "clean doc two",
    ]
    report = DecontaminationReport()
    kept = list(decontaminate_iter(docs, idx, drop_contaminated=False, report=report))
    assert kept == docs
    assert report.n_corpus_docs_with_any_match == 1


def test_decontaminate_iter_accepts_dict_docs():
    """Custom text_fn lets callers pass Document objects."""
    idx = DecontaminationIndex(DecontaminationConfig(ngram_size=4))
    idx.add_benchmark("bench", ["the capital of france is paris"])
    docs = [
        {"id": "d1", "text": "clean"},
        {"id": "d2", "text": "the capital of france is paris yes"},
    ]
    kept = list(decontaminate_iter(
        docs, idx,
        text_fn=lambda d: d["text"],
        id_fn=lambda d: d["id"],
    ))
    assert len(kept) == 1
    assert kept[0]["id"] == "d1"


# --------------------------------------------------------------------------- #
# Edge cases
# --------------------------------------------------------------------------- #
def test_empty_corpus():
    idx = DecontaminationIndex(DecontaminationConfig(ngram_size=4))
    idx.add_benchmark("bench", ["alpha beta gamma delta"])
    report = DecontaminationReport()
    kept = list(decontaminate_iter([], idx, report=report))
    assert kept == []
    assert report.n_corpus_docs_scanned == 0


def test_empty_benchmark():
    idx = DecontaminationIndex(DecontaminationConfig(ngram_size=4))
    idx.add_benchmark("bench", [])
    matches = idx.scan_document("anything at all here yes")
    assert matches == {}
