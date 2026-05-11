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

    # --------------------------------------------------------------- #
    # P0-6 fix (2026-05-12 audit): MixtureSampler must achieve TOKEN
    # shares matching the yaml weights, not document shares. Before the
    # fix, picking one doc per step with weights as doc-probabilities
    # silently inverted intended token mixes when doc lengths varied
    # widely between sources (pg19 ~30K tok/doc vs FineWeb-Edu ~500
    # tok/doc → tiny doc-share for pg19 = huge token-share).
    # --------------------------------------------------------------- #
    def test_token_share_matches_target_when_doc_lengths_differ(self):
        """Source 'short' has 1-char docs; source 'long' has 100-char docs.
        With target weights 0.5/0.5 the OLD impl would emit 50/50 docs =
        ~1% from 'short', ~99% from 'long' by character count. The fixed
        impl should self-correct via deficit-driven reweighting and
        emit ~50/50 by character count."""
        # Big pool of each so we don't exhaust mid-run.
        sampler = MixtureSampler(
            sources={
                "short": iter(["x"] * 50000),  # 1 char per "doc"
                "long": iter(["y" * 100] * 50000),  # 100 chars per "doc"
            },
            weights=[SourceWeight("short", 0.5), SourceWeight("long", 0.5)],
            seed=7,
            on_exhaust="stop",
            bootstrap_steps=32,
        )
        emitted_chars = {"short": 0, "long": 0}
        for i, (src, ex) in enumerate(sampler):
            emitted_chars[src] += len(ex)
            if i >= 10000:
                break

        total = sum(emitted_chars.values())
        share_short = emitted_chars["short"] / total
        # Target is 0.5, observed should be within ±5% (deficit-driven
        # convergence is fast but not perfect; bootstrap noise lasts
        # ~32 picks).
        assert 0.45 < share_short < 0.55, (
            f"token-share for 'short' should be ~0.5, got {share_short:.3f}. "
            f"If <0.1, the MixtureSampler is back to doc-weighted (P0-6 regression)."
        )

    def test_token_share_matches_target_with_skewed_weights(self):
        """Target weights 0.8 short, 0.2 long. With doc lengths 1 and 100,
        the new impl should achieve token-share ≈ 0.8/0.2 not the wrong
        doc-share."""
        sampler = MixtureSampler(
            sources={
                "short": iter(["x"] * 100000),
                "long": iter(["y" * 100] * 100000),
            },
            weights=[SourceWeight("short", 0.8), SourceWeight("long", 0.2)],
            seed=11,
            on_exhaust="stop",
            bootstrap_steps=32,
        )
        emitted_chars = {"short": 0, "long": 0}
        for i, (src, ex) in enumerate(sampler):
            emitted_chars[src] += len(ex)
            if i >= 20000:
                break

        total = sum(emitted_chars.values())
        share_short = emitted_chars["short"] / total
        assert 0.75 < share_short < 0.85, (
            f"token-share for 'short' should be ~0.8, got {share_short:.3f}"
        )

    def test_emitted_per_source_introspection(self):
        """The sampler exposes emitted_per_source for telemetry."""
        sampler = MixtureSampler(
            sources={"a": iter(["aa"] * 100), "b": iter(["bbbb"] * 100)},
            weights=[SourceWeight("a", 0.5), SourceWeight("b", 0.5)],
            seed=0,
            on_exhaust="stop",
        )
        for i, _ in enumerate(sampler):
            if i >= 200:
                break
        # After iteration, the dict should have non-zero counts.
        assert sampler.emitted_per_source["a"] > 0
        assert sampler.emitted_per_source["b"] > 0


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
        assert all(len(s.tokens) == 8 for s in out)
        assert out[0].tokens == [1, 2, 3, 99, 4, 5, 6, 7]

    def test_pad_when_drop_last_false(self):
        packer = SequencePacker(
            sequence_length=8, eos_token_id=99, drop_last=False, pad_token_id=0
        )
        out = list(packer.pack([[1, 2, 3]]))
        assert len(out) == 1
        # 3 tokens + 1 EOS = 4, padded to 8 with pad_id=0
        assert out[0].tokens == [1, 2, 3, 99, 0, 0, 0, 0]

    # ----------------------------------------------------------------- #
    # P0-2 fix (2026-05-12 audit): segment_ids must be emitted alongside
    # tokens so the downstream attention mask + loss mask can use them.
    # Before this fix the packer returned bare token lists and the
    # documentation falsely claimed intra-doc masking was active.
    # ----------------------------------------------------------------- #
    def test_segment_ids_increment_at_document_boundaries(self):
        packer = SequencePacker(sequence_length=8, eos_token_id=99, drop_last=True)
        out = list(
            packer.pack([
                [1, 2, 3],
                [4, 5, 6, 7, 8],
                [9, 10, 11, 12, 13, 14, 15],
            ])
        )
        # First packed seq is [1,2,3, 99, 4,5,6,7]
        # doc 0 = [1,2,3,99],  doc 1 starts at index 4.
        assert out[0].segment_ids == [0, 0, 0, 0, 1, 1, 1, 1]
        # Second packed seq is [8, 99, 9, 10, 11, 12, 13, 14]
        # doc 1's tail (8 + EOS), then doc 2 starts.
        assert out[1].segment_ids == [1, 1, 2, 2, 2, 2, 2, 2]

    def test_segment_ids_length_matches_tokens(self):
        packer = SequencePacker(sequence_length=8, eos_token_id=99, drop_last=True)
        out = list(packer.pack([[1, 2, 3], [4, 5, 6, 7, 8]]))
        for ps in out:
            assert len(ps.tokens) == len(ps.segment_ids)

    def test_pad_positions_get_sentinel_segment_id(self):
        """Padding tokens must NOT share a segment with any real document.
        Use sentinel -1 so downstream loss mask drops them."""
        packer = SequencePacker(
            sequence_length=8, eos_token_id=99, drop_last=False, pad_token_id=0
        )
        out = list(packer.pack([[1, 2, 3]]))
        # 3 real tokens + 1 EOS = 4 → 4 pad positions
        assert out[0].segment_ids[:4] == [0, 0, 0, 0]  # doc 0 (incl EOS)
        assert out[0].segment_ids[4:] == [-1, -1, -1, -1]  # padding sentinel


class TestMakeInputLabelPairs:
    """The pairs maker is the place where document-boundary loss masking
    happens. P0-2 audit fix added segment_ids + loss_mask to the output."""

    def test_emits_4_tuple_with_loss_mask_at_doc_boundaries(self):
        from myllm.data.pack import PackedSequence
        from myllm.data.tokenize import make_input_label_pairs

        # Manually crafted packed sequence: doc 0 = [10, 11, 99] (EOS),
        # doc 1 = [20, 21, 22, 99]. segment_ids: [0,0,0, 1,1,1,1].
        # After shift: input[:-1] = [10,11,99, 20,21,22], label[1:] = [11,99, 20,21,22,99]
        # input_segments = [0,0,0, 1,1,1], label_segments = [0,0, 1,1,1,1]
        # loss_mask = [in_seg == lab_seg AND in_seg != -1]
        #           = [1,  1,  0,  1, 1, 1]
        ps = PackedSequence(
            tokens=[10, 11, 99, 20, 21, 22, 99],
            segment_ids=[0, 0, 0, 1, 1, 1, 1],
        )
        out = list(make_input_label_pairs([ps]))
        assert len(out) == 1
        inp, lab, seg, mask = out[0]
        assert inp == [10, 11, 99, 20, 21, 22]
        assert lab == [11, 99, 20, 21, 22, 99]
        assert seg == [0, 0, 0, 1, 1, 1]
        # Position 2: input is EOS of doc 0, label is first token of doc 1.
        # Predicting doc 1's content from doc 0's EOS is meaningless → mask 0.
        assert mask == [1, 1, 0, 1, 1, 1]

    def test_pad_positions_get_loss_mask_zero(self):
        from myllm.data.pack import PackedSequence
        from myllm.data.tokenize import make_input_label_pairs

        # Doc 0 = [10, 11, 99], padded to length 5 with [0, 0]
        # segment_ids = [0, 0, 0, -1, -1]
        ps = PackedSequence(
            tokens=[10, 11, 99, 0, 0],
            segment_ids=[0, 0, 0, -1, -1],
        )
        out = list(make_input_label_pairs([ps]))
        inp, lab, seg, mask = out[0]
        # input_segments = [0, 0, 0, -1], label_segments = [0, 0, -1, -1]
        # mask: pos 0=(0==0,!=-1)=1; pos 1=(0==0,!=-1)=1; pos 2=(0!=-1 but -1 fails)=0;
        #       pos 3=(-1==-1 but -1 fails)=0
        assert mask == [1, 1, 0, 0]

    def test_back_compat_with_bare_token_lists(self):
        """Old callers passing bare list[int] still work — segment_ids
        defaults to all-zero (one segment per pack), loss_mask all-one."""
        from myllm.data.tokenize import make_input_label_pairs

        out = list(make_input_label_pairs([[1, 2, 3, 4, 5]]))
        assert len(out) == 1
        inp, lab, seg, mask = out[0]
        assert inp == [1, 2, 3, 4]
        assert lab == [2, 3, 4, 5]
        assert seg == [0, 0, 0, 0]
        assert mask == [1, 1, 1, 1]


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
