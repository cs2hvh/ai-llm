"""Regression tests for the decontamination pipeline integration (R6 P0).

Covers:
  - ``DecontaminationFilter`` keeps clean docs, rejects contaminated docs,
    populates the optional report, and threads cleanly through ``FilterChain``.
  - ``extract_prompts_from_benchmark`` pulls ``.prompt`` from EvalExample-
    shaped objects and dict-shaped examples; raises on unsupported shapes.
  - ``index_from_benchmarks`` uses ``benchmark.name`` as the id and combines
    multiple benchmarks correctly.
  - ``DecontaminationIndex.save_json`` / ``load_json`` round-trip and the
    reverse-lookup map is rebuilt on load (so ``scan_document`` still works).
  - End-to-end: a ``FilterChain`` with decon + PII redacts surviving docs.

The point of these tests is the *wiring* — the core scan logic is already
covered by ``test_decontamination.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from myllm.data.decontamination import (
    DecontaminationConfig,
    DecontaminationIndex,
    DecontaminationReport,
    extract_prompts_from_benchmark,
    index_from_benchmarks,
)
from myllm.data.filters import (
    DecontaminationFilter,
    FilterChain,
    LengthFilter,
    PIIRedactor,
)
from myllm.data.types import Document


def _doc(text: str, doc_id: str = "d1") -> Document:
    return Document(text=text, source="web", dataset="test", doc_id=doc_id)


# --------------------------------------------------------------------------- #
# DecontaminationFilter
# --------------------------------------------------------------------------- #
class TestDecontaminationFilter:
    def test_keeps_clean_doc(self):
        idx = DecontaminationIndex(DecontaminationConfig(ngram_size=4))
        idx.add_benchmark("bench", ["the capital of france is paris"])
        f = DecontaminationFilter(index=idx)
        decision = f.apply(_doc("totally unrelated content here yes indeed"))
        assert decision.keep is True
        assert decision.reason == "ok"

    def test_rejects_contaminated_doc(self):
        idx = DecontaminationIndex(DecontaminationConfig(ngram_size=4))
        idx.add_benchmark("bench", ["the capital of france is paris"])
        f = DecontaminationFilter(index=idx)
        decision = f.apply(_doc("Yes, the capital of france is paris and that's that."))
        assert decision.keep is False
        assert decision.reason == "contaminated"
        # score = total n-gram matches across all benchmarks. With ngram=4
        # and a 6-word verbatim phrase, we expect 6-4+1=3 sliding hits.
        assert decision.score >= 3

    def test_populates_report(self):
        idx = DecontaminationIndex(DecontaminationConfig(ngram_size=4))
        idx.add_benchmark("bench", ["alpha beta gamma delta epsilon zeta"])
        report = DecontaminationReport()
        f = DecontaminationFilter(index=idx, report=report)

        f.apply(_doc("alpha beta gamma delta epsilon zeta tail"))  # hit
        f.apply(_doc("totally clean and not related"))             # clean
        f.apply(_doc("again: alpha beta gamma delta epsilon zeta")) # hit

        assert report.n_corpus_docs_scanned == 3
        assert report.n_corpus_docs_with_any_match == 2
        assert report.per_bench["bench"]["n_corpus_docs_matched"] == 2

    def test_doc_id_threaded_into_report(self):
        idx = DecontaminationIndex(DecontaminationConfig(ngram_size=4))
        idx.add_benchmark("bench", ["alpha beta gamma delta epsilon"])
        report = DecontaminationReport()
        f = DecontaminationFilter(index=idx, report=report)
        f.apply(_doc("alpha beta gamma delta epsilon end", doc_id="doc-007"))
        examples = report.per_bench["bench"]["example_matches"]
        # The doc_id should be captured in the report's per-bench example list.
        assert any(e.get("doc_id") == "doc-007" for e in examples)

    def test_empty_index_is_pass_through(self):
        """An index with no benchmarks rejects nothing — useful for staged
        rollout where you've enabled the filter but not yet loaded prompts."""
        idx = DecontaminationIndex(DecontaminationConfig(ngram_size=4))
        f = DecontaminationFilter(index=idx)
        decision = f.apply(_doc("anything at all here yes please"))
        assert decision.keep is True
        assert decision.reason == "ok"


# --------------------------------------------------------------------------- #
# Chain integration
# --------------------------------------------------------------------------- #
class TestFilterChainWithDecontamination:
    def test_chain_rejects_contaminated_before_pii(self):
        """The DecontaminationFilter should reject before PIIRedactor even
        runs, so contaminated docs don't get redacted unnecessarily."""
        idx = DecontaminationIndex(DecontaminationConfig(ngram_size=4))
        idx.add_benchmark("bench", ["the capital of france is paris"])
        chain = FilterChain(
            (
                LengthFilter(min_chars=10),
                DecontaminationFilter(index=idx),
                PIIRedactor(),
            )
        )
        doc = _doc("Email me at x@y.com — also the capital of france is paris.")
        out, decision = chain.apply(doc)
        assert decision.keep is False
        assert decision.reason == "contaminated"
        # PIIRedactor should NOT have run — the doc text still has the email.
        assert "x@y.com" in out.text

    def test_chain_keeps_clean_doc_and_redacts_pii(self):
        idx = DecontaminationIndex(DecontaminationConfig(ngram_size=4))
        idx.add_benchmark("bench", ["the capital of france is paris"])
        chain = FilterChain(
            (
                LengthFilter(min_chars=10),
                DecontaminationFilter(index=idx),
                PIIRedactor(),
            )
        )
        # Clean: passes decon, then PII redaction fires.
        doc = _doc("Reach me at user@example.com for follow-up details please.")
        out, decision = chain.apply(doc)
        assert decision.keep is True
        assert "user@example.com" not in out.text
        assert "<EMAIL>" in out.text

    def test_short_doc_rejected_before_decon_runs(self):
        """LengthFilter is upstream of the decon filter; a too-short doc
        should not even reach the decon filter."""
        idx = DecontaminationIndex(DecontaminationConfig(ngram_size=4))
        idx.add_benchmark("bench", ["the capital of france is paris"])
        report = DecontaminationReport()
        chain = FilterChain(
            (
                LengthFilter(min_chars=200),
                DecontaminationFilter(index=idx, report=report),
            )
        )
        # Short doc that *would* be flagged as contaminated if reached.
        out, decision = chain.apply(_doc("the capital of france is paris."))
        assert decision.keep is False
        assert decision.reason == "too_short"
        # The decon filter never ran → report is empty.
        assert report.n_corpus_docs_scanned == 0


# --------------------------------------------------------------------------- #
# Benchmark-adapter bridge
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _FakeExample:
    """Stand-in for EvalExample for tests — we don't import the real one
    here to keep this test file independent of the eval package."""

    prompt: str
    target_answer: str = "A"
    metadata: dict = field(default_factory=dict)


class _FakeBenchmark:
    """Minimal Benchmark stub yielding fixed prompts."""

    def __init__(self, name: str, prompts: list[str]):
        self.name = name
        self._prompts = prompts

    def load_examples(self, split: str = "test", sample_size: int | None = None, seed: int = 0):
        examples = [_FakeExample(prompt=p) for p in self._prompts]
        if sample_size is not None:
            examples = examples[:sample_size]
        return iter(examples)


class TestExtractPromptsFromBenchmark:
    def test_extracts_prompt_attribute(self):
        bench = _FakeBenchmark("fake", ["alpha beta", "gamma delta"])
        prompts = list(extract_prompts_from_benchmark(bench))
        assert prompts == ["alpha beta", "gamma delta"]

    def test_extracts_dict_prompt_key(self):
        class DictBench:
            name = "dict-bench"

            def load_examples(self, split="test", sample_size=None, seed=0):
                yield {"prompt": "first prompt", "target_answer": "A"}
                yield {"prompt": "second prompt", "target_answer": "B"}

        prompts = list(extract_prompts_from_benchmark(DictBench()))
        assert prompts == ["first prompt", "second prompt"]

    def test_raises_on_unsupported_shape(self):
        class WeirdBench:
            name = "weird"

            def load_examples(self, split="test", sample_size=None, seed=0):
                yield "raw string not a dict not an object"

        with pytest.raises(TypeError, match="neither.*attribute nor.*key"):
            list(extract_prompts_from_benchmark(WeirdBench()))

    def test_sample_size_is_threaded_through(self):
        bench = _FakeBenchmark("fake", ["a", "b", "c", "d", "e"])
        prompts = list(extract_prompts_from_benchmark(bench, sample_size=3))
        assert prompts == ["a", "b", "c"]


# --------------------------------------------------------------------------- #
# index_from_benchmarks convenience builder
# --------------------------------------------------------------------------- #
class TestIndexFromBenchmarks:
    def test_uses_benchmark_name_as_id(self):
        b1 = _FakeBenchmark("alpha-bench", ["one two three four five six"])
        b2 = _FakeBenchmark("beta-bench", ["seven eight nine ten eleven twelve"])
        idx = index_from_benchmarks([b1, b2], config=DecontaminationConfig(ngram_size=4))
        assert set(idx.signatures.keys()) == {"alpha-bench", "beta-bench"}

    def test_multi_benchmark_attribution(self):
        b1 = _FakeBenchmark("alpha-bench", ["aaa bbb ccc ddd eee fff"])
        b2 = _FakeBenchmark("beta-bench", ["xxx yyy zzz www vvv uuu"])
        idx = index_from_benchmarks([b1, b2], config=DecontaminationConfig(ngram_size=4))
        m = idx.scan_document("text aaa bbb ccc ddd eee fff and xxx yyy zzz www vvv uuu end")
        assert "alpha-bench" in m
        assert "beta-bench" in m


# --------------------------------------------------------------------------- #
# JSON round-trip
# --------------------------------------------------------------------------- #
class TestIndexSerialization:
    def test_save_load_round_trip(self, tmp_path):
        idx = DecontaminationIndex(DecontaminationConfig(ngram_size=4, hash_seed=42))
        idx.add_benchmark("bench-a", ["alpha beta gamma delta epsilon"])
        idx.add_benchmark("bench-b", ["one two three four five six"])
        path = tmp_path / "decon.json"
        idx.save_json(path)

        loaded = DecontaminationIndex.load_json(path)
        # Config preserved.
        assert loaded.config.ngram_size == 4
        assert loaded.config.hash_seed == 42
        # Signatures preserved.
        assert set(loaded.signatures.keys()) == {"bench-a", "bench-b"}
        assert loaded.signatures["bench-a"].ngrams == idx.signatures["bench-a"].ngrams
        # Reverse-lookup map rebuilt — scan behaves identically.
        m_orig = idx.scan_document("text alpha beta gamma delta epsilon tail")
        m_loaded = loaded.scan_document("text alpha beta gamma delta epsilon tail")
        assert m_orig == m_loaded

    def test_loaded_index_supports_filter(self, tmp_path):
        """Wire the load path through the actual filter to catch any
        cross-cutting integration bugs."""
        idx = DecontaminationIndex(DecontaminationConfig(ngram_size=4))
        idx.add_benchmark("bench", ["the capital of france is paris"])
        path = tmp_path / "decon.json"
        idx.save_json(path)

        loaded = DecontaminationIndex.load_json(path)
        f = DecontaminationFilter(index=loaded)
        # Same doc that the in-memory filter rejected:
        decision = f.apply(_doc("Yes, the capital of france is paris and that's that."))
        assert decision.keep is False
        assert decision.reason == "contaminated"

    def test_n_examples_preserved(self, tmp_path):
        idx = DecontaminationIndex(DecontaminationConfig(ngram_size=4))
        idx.add_benchmark("bench", ["a b c d e", "f g h i j", "k l m n o"])
        path = tmp_path / "decon.json"
        idx.save_json(path)
        loaded = DecontaminationIndex.load_json(path)
        assert loaded.signatures["bench"].n_examples == 3


# --------------------------------------------------------------------------- #
# Build path from run_pretrain
# --------------------------------------------------------------------------- #
class TestRunPretrainBuilder:
    def test_disabled_decon_returns_none(self):
        """When the yaml has no decontamination block, the builder returns
        None and the filter chain skips the decon filter entirely."""
        import sys
        from pathlib import Path as _P
        sys.path.insert(0, str(_P("/root/llm-build/scripts").resolve()))
        from run_pretrain import build_decontamination_index

        assert build_decontamination_index(None) is None
        assert build_decontamination_index({}) is None
        assert build_decontamination_index({"enabled": False}) is None

    def test_loads_prebuilt_index(self, tmp_path):
        import sys
        from pathlib import Path as _P
        sys.path.insert(0, str(_P("/root/llm-build/scripts").resolve()))
        from run_pretrain import build_decontamination_index

        # Build + save an index.
        idx = DecontaminationIndex(DecontaminationConfig(ngram_size=4))
        idx.add_benchmark("test-bench", ["one two three four five"])
        path = tmp_path / "decon.json"
        idx.save_json(path)

        loaded = build_decontamination_index(
            {"enabled": True, "index_path": str(path)}
        )
        assert loaded is not None
        assert "test-bench" in loaded.signatures

    def test_enabled_without_path_or_benchmarks_raises(self):
        import sys
        from pathlib import Path as _P
        sys.path.insert(0, str(_P("/root/llm-build/scripts").resolve()))
        from run_pretrain import build_decontamination_index

        with pytest.raises(ValueError, match="index_path.*benchmarks"):
            build_decontamination_index({"enabled": True})
