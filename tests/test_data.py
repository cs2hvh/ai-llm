"""Unit tests for the data pipeline (pure-Python, no GPU)."""
from __future__ import annotations

import pytest

from myllm.data import (
    Document,
    FilterChain,
    LengthFilter,
    MixtureSampler,
    PIIRedactor,
    RepetitionFilter,
    SequencePacker,
    SourceWeight,
    SymbolRatioFilter,
)


def make_doc(text: str, **kw) -> Document:
    return Document(
        text=text,
        source=kw.get("source", "web"),
        dataset=kw.get("dataset", "test/ds"),
        doc_id=kw.get("doc_id", "1"),
    )


# --------------------------------------------------------------------------- #
# Filters
# --------------------------------------------------------------------------- #
class TestLengthFilter:
    def test_too_short_rejected(self):
        f = LengthFilter(min_chars=10)
        d = f.apply(make_doc("hi"))
        assert not d.keep and d.reason == "too_short"

    def test_too_long_rejected(self):
        f = LengthFilter(min_chars=1, max_chars=5)
        d = f.apply(make_doc("hello world"))
        assert not d.keep and d.reason == "too_long"

    def test_in_range_kept(self):
        f = LengthFilter(min_chars=2, max_chars=100)
        d = f.apply(make_doc("hello"))
        assert d.keep

    def test_invalid_bounds_raises(self):
        with pytest.raises(ValueError):
            LengthFilter(min_chars=10, max_chars=5)


class TestRepetitionFilter:
    def test_top_word_repeat_rejected(self):
        f = RepetitionFilter(max_top_word_share=0.20, ngram_n=5)
        # 80% the same word
        text = "spam " * 80 + " ".join(f"w{i}" for i in range(20))
        d = f.apply(make_doc(text))
        assert not d.keep and d.reason == "top_word_repeated"

    def test_normal_text_kept(self):
        f = RepetitionFilter()
        text = " ".join(f"unique{i}" for i in range(100))
        d = f.apply(make_doc(text))
        assert d.keep


class TestSymbolRatioFilter:
    def test_symbolic_rejected(self):
        f = SymbolRatioFilter(max_symbol_ratio=0.30)
        d = f.apply(make_doc("!@#$%^&*()" * 10))
        assert not d.keep

    def test_text_kept(self):
        f = SymbolRatioFilter()
        d = f.apply(make_doc("Hello world, this is a normal sentence."))
        assert d.keep


class TestPIIRedactor:
    def test_email_redacted(self):
        f = PIIRedactor()
        doc = make_doc("Contact me at alice@example.com please")
        d = f.apply(doc)
        assert d.keep
        assert "<EMAIL>" in doc.text
        assert "alice@example.com" not in doc.text

    def test_phone_redacted(self):
        f = PIIRedactor()
        doc = make_doc("Call 555-123-4567 anytime")
        d = f.apply(doc)
        assert "<PHONE>" in doc.text


class TestFilterChain:
    def test_short_circuits_on_reject(self):
        chain = FilterChain((LengthFilter(min_chars=100), PIIRedactor()))
        doc = make_doc("hi alice@x.com")
        out, decision = chain.apply(doc)
        assert not decision.keep and decision.reason == "too_short"
        # Redaction did NOT run because length rejected first.
        assert "alice@x.com" in out.text

    def test_all_pass(self):
        chain = FilterChain((LengthFilter(min_chars=2), PIIRedactor()))
        doc = make_doc("contact alice@x.com")
        _, decision = chain.apply(doc)
        assert decision.keep


# --------------------------------------------------------------------------- #
# Mixture sampler
# --------------------------------------------------------------------------- #
class TestMixtureSampler:
    def test_weights_respected_in_expectation(self):
        sampler = MixtureSampler(
            sources={"a": iter(["a"] * 10000), "b": iter(["b"] * 10000)},
            weights=[SourceWeight("a", 0.7), SourceWeight("b", 0.3)],
            seed=42,
            on_exhaust="stop",
        )
        counts = {"a": 0, "b": 0}
        for i, (src, _) in enumerate(sampler):
            counts[src] += 1
            if i >= 5000:
                break
        ratio_a = counts["a"] / sum(counts.values())
        assert 0.65 < ratio_a < 0.75  # within ~5%

    def test_drop_on_exhaust_continues(self):
        sampler = MixtureSampler(
            sources={"a": iter(["a"] * 5), "b": iter(["b"] * 100)},
            weights=[SourceWeight("a", 0.5), SourceWeight("b", 0.5)],
            seed=0,
            on_exhaust="drop",
        )
        items = list(sampler)
        # All 5 'a's plus all 100 'b's should be drawn (order varies).
        assert sum(1 for s, _ in items if s == "a") == 5
        assert sum(1 for s, _ in items if s == "b") == 100

    def test_unknown_source_rejected(self):
        with pytest.raises(Exception):
            MixtureSampler(
                sources={"a": iter([])},
                weights=[SourceWeight("nonexistent", 1.0)],
            )


# --------------------------------------------------------------------------- #
# Sequence packer
# --------------------------------------------------------------------------- #
class TestSequencePacker:
    def test_packs_into_fixed_length(self):
        packer = SequencePacker(sequence_length=8, eos_token_id=99, drop_last=True)
        # Three docs of length 3, 5, 7 → with EOS = 4, 6, 8 → total 18 tokens
        # → 2 sequences of length 8, drop the last 2 tokens.
        out = list(
            packer.pack([
                [1, 2, 3],
                [4, 5, 6, 7, 8],
                [9, 10, 11, 12, 13, 14, 15],
            ])
        )
        assert len(out) == 2
        assert all(len(s) == 8 for s in out)
        assert out[0] == [1, 2, 3, 99, 4, 5, 6, 7]

    def test_pad_when_drop_last_false(self):
        packer = SequencePacker(
            sequence_length=8, eos_token_id=99, drop_last=False, pad_token_id=0
        )
        out = list(packer.pack([[1, 2, 3]]))
        assert len(out) == 1
        # 3 tokens + 1 EOS = 4, padded to 8 with pad_id=0
        assert out[0] == [1, 2, 3, 99, 0, 0, 0, 0]


# --------------------------------------------------------------------------- #
# Dedupe
# --------------------------------------------------------------------------- #
class TestDeduplicator:
    def test_exact_duplicate_detected(self):
        xxhash = pytest.importorskip("xxhash")  # noqa: F841
        from myllm.data.dedupe import Deduplicator, MinHashConfig

        d = Deduplicator(MinHashConfig(num_perm=64, num_bands=16, ngram_size=3, threshold=0.9))
        text = "the quick brown fox jumps over the lazy dog and runs into the woods"
        added1, _ = d.add_if_new("a", text)
        added2, dup_of = d.add_if_new("b", text)
        assert added1
        assert not added2
        assert dup_of == "a"

    def test_distinct_docs_kept(self):
        xxhash = pytest.importorskip("xxhash")  # noqa: F841
        from myllm.data.dedupe import Deduplicator, MinHashConfig

        d = Deduplicator(MinHashConfig(num_perm=64, num_bands=16, ngram_size=3, threshold=0.9))
        added1, _ = d.add_if_new("a", "the quick brown fox jumps over the lazy dog")
        added2, _ = d.add_if_new(
            "b", "completely different text about machine learning research"
        )
        assert added1 and added2

    def test_invalid_config(self):
        from myllm.data.dedupe import MinHashConfig

        with pytest.raises(ValueError):
            MinHashConfig(num_perm=128, num_bands=33)  # 128 not divisible by 33
