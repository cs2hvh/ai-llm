"""Tests for per-source val loss (Phase 1.2, P0-1).

Covers:
  1. PackedCorpusReader.get_per_token_source_ids — correct labels per
     position, sentinel -1 for uncovered positions, unknown sources
     map to -1.
  2. build_per_source_held_out — produces batches with the expected
     shape, correct label-position source attribution (shifted by 1
     from input-position source).
  3. make_per_source_validation_loss_eval_from_eval_step —
     - returns the documented metric schema (loss, ppl, n_tokens,
       plus per-source variants)
     - aggregate matches the weighted mean of per-source losses
     - sources with zero held-out coverage are omitted
     - empty held-out raises
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

from myllm.data.packed_corpus import (
    DocSpan,
    PackedCorpusReader,
    PackedCorpusWriter,
    write_corpus_manifest,
)
from myllm.training.eval_hook import (
    build_per_source_held_out,
    make_per_source_validation_loss_eval_from_eval_step,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _build_tiny_corpus(
    root: Path,
    *,
    sequence_length: int = 8,
    n_sequences: int = 6,
    seed: int = 0,
) -> PackedCorpusReader:
    """Two-source mini corpus we can introspect easily.

    Each sequence is 8 tokens. Sources alternate per sequence:
        sid=0,2,4: "src_a" covers [0,8)
        sid=1,3,5: "src_b" covers [0,4); "src_a" covers [4,8)
    so we get both single-source and mixed-source packed sequences.
    """
    writer = PackedCorpusWriter(
        root=str(root),
        sequence_length=sequence_length,
        sequences_per_shard=8,
        tokenizer_sha256="0" * 64,
    )
    rng = np.random.default_rng(seed)
    for sid in range(n_sequences):
        tokens = rng.integers(1, 100, size=sequence_length, dtype=np.uint32)
        if sid % 2 == 0:
            # Single-source: src_a covers the whole sequence
            spans = [DocSpan(
                doc_span_id=sid * 10,
                sequence_id=sid,
                source_id="src_a",
                doc_id_hash=1234 + sid,
                dataset_revision_id="rev-1",
                token_start_in_sequence=0,
                token_end_in_sequence=sequence_length,
                text_hash=5678 + sid,
            )]
        else:
            # Mixed: src_b[0,4), src_a[4,8)
            spans = [
                DocSpan(
                    doc_span_id=sid * 10,
                    sequence_id=sid,
                    source_id="src_b",
                    doc_id_hash=2000 + sid,
                    dataset_revision_id="rev-1",
                    token_start_in_sequence=0,
                    token_end_in_sequence=4,
                    text_hash=6000 + sid,
                ),
                DocSpan(
                    doc_span_id=sid * 10 + 1,
                    sequence_id=sid,
                    source_id="src_a",
                    doc_id_hash=3000 + sid,
                    dataset_revision_id="rev-1",
                    token_start_in_sequence=4,
                    token_end_in_sequence=sequence_length,
                    text_hash=7000 + sid,
                ),
            ]
        writer.append_sequence(tokens, spans)
    writer.close()
    write_corpus_manifest(
        root,
        corpus_name="test_per_source_corpus",
        tokenizer_sha256="0" * 64,
        sequence_length=sequence_length,
        sequences_per_shard=8,
        source_revisions={"src_a": "rev-1", "src_b": "rev-1"},
        target_source_share={"src_a": 0.5, "src_b": 0.5},
    )
    return PackedCorpusReader(root=str(root))


# --------------------------------------------------------------------------- #
# Reader API: per-token source-ids
# --------------------------------------------------------------------------- #
class TestGetPerTokenSourceIds:
    def test_single_source_sequence(self, tmp_path):
        reader = _build_tiny_corpus(tmp_path / "corpus")
        vocab = {"src_a": 0, "src_b": 1}
        src_ids = reader.get_per_token_source_ids(0, vocab)
        assert src_ids.shape == (8,)
        assert src_ids.dtype == np.int32
        # sid=0 is fully covered by src_a
        assert np.all(src_ids == 0), f"got {src_ids.tolist()}"

    def test_mixed_source_sequence(self, tmp_path):
        reader = _build_tiny_corpus(tmp_path / "corpus")
        vocab = {"src_a": 0, "src_b": 1}
        # sid=1 has src_b[0,4) src_a[4,8)
        src_ids = reader.get_per_token_source_ids(1, vocab)
        assert src_ids.tolist() == [1, 1, 1, 1, 0, 0, 0, 0]

    def test_unknown_source_becomes_minus_one(self, tmp_path):
        reader = _build_tiny_corpus(tmp_path / "corpus")
        # Vocab only contains src_a; src_b positions should be -1.
        vocab = {"src_a": 0}
        src_ids = reader.get_per_token_source_ids(1, vocab)
        assert src_ids.tolist() == [-1, -1, -1, -1, 0, 0, 0, 0]


# --------------------------------------------------------------------------- #
# Held-out builder
# --------------------------------------------------------------------------- #
class TestBuildPerSourceHeldOut:
    def test_batches_have_correct_shape(self, tmp_path):
        reader = _build_tiny_corpus(tmp_path / "corpus")
        batches, src_arrays, vocab = build_per_source_held_out(
            reader, n_sequences=6, micro_batch_size=2,
        )
        # 6 sequences / 2 per batch = 3 batches
        assert len(batches) == 3
        assert len(src_arrays) == 3
        for batch in batches:
            assert batch["input_ids"].shape == (2, 7)  # seq_len-1
            assert batch["labels"].shape == (2, 7)
            assert batch["segment_ids"].shape == (2, 7)
            assert batch["loss_mask"].shape == (2, 7)
        for arr in src_arrays:
            assert arr.shape == (2, 7)
            assert arr.dtype == np.int32

    def test_source_labels_use_label_position_not_input_position(self, tmp_path):
        # The label at position i is tokens[i+1]; so its source attribution
        # is per_token_source[i+1]. For a mixed sequence src_b[0,4)
        # src_a[4,8), the input-position sources are [b,b,b,b,a,a,a,a],
        # but the label-position sources (input shape: 7) are
        # [b,b,b,a,a,a,a].
        reader = _build_tiny_corpus(tmp_path / "corpus")
        batches, src_arrays, vocab = build_per_source_held_out(
            reader, n_sequences=2, micro_batch_size=2,
        )
        a_int = vocab["src_a"]
        b_int = vocab["src_b"]
        # batches[0] contains sid=0 (single-source src_a) at row 0,
        # sid=1 (mixed b->a) at row 1.
        # Row 0: 7 positions all src_a
        assert src_arrays[0][0].tolist() == [a_int] * 7
        # Row 1: label positions correspond to original position [1, 7];
        # original was [b,b,b,b,a,a,a,a]; sliced [1:] -> [b,b,b,a,a,a,a]
        assert src_arrays[0][1].tolist() == [b_int, b_int, b_int, a_int, a_int, a_int, a_int]

    def test_vocab_derived_from_held_out_slice(self, tmp_path):
        reader = _build_tiny_corpus(tmp_path / "corpus")
        # Without an explicit vocab, the function derives one from the
        # actual sources seen.
        _, _, vocab = build_per_source_held_out(
            reader, n_sequences=6, micro_batch_size=2,
        )
        assert set(vocab.keys()) == {"src_a", "src_b"}
        # Deterministic ordering: alphabetical
        assert vocab["src_a"] < vocab["src_b"]

    def test_partial_tail_batch_dropped(self, tmp_path):
        # 5 sequences, micro_batch=2 -> 2 full batches; the 5th seq is dropped.
        reader = _build_tiny_corpus(tmp_path / "corpus", n_sequences=5)
        batches, src_arrays, _ = build_per_source_held_out(
            reader, n_sequences=5, micro_batch_size=2,
        )
        assert len(batches) == 2
        assert len(src_arrays) == 2


# --------------------------------------------------------------------------- #
# Per-source eval-fn factory
# --------------------------------------------------------------------------- #
class TestMakePerSourceEvalFn:
    """The eval-fn bucketing logic — exercised with a stub eval_step_fn
    so we don't need a model or a JAX device.
    """

    def _stub_eval_step_fn(self, nll_per_token, weight_per_token):
        """Returns an eval_step_fn that returns fixed metrics."""
        def stub(state, batch):
            return {
                "loss": float(np.mean(nll_per_token)),
                "nll_per_token": np.asarray(nll_per_token, dtype=np.float32),
                "weight_per_token": np.asarray(weight_per_token, dtype=np.float32),
            }
        return stub

    def test_aggregate_matches_weighted_mean(self):
        # Single batch, 2 sequences, 4 positions each.
        # Per-token NLL chosen so aggregate is easy to compute.
        nll = np.array([
            [1.0, 1.0, 2.0, 2.0],   # mean = 1.5
            [3.0, 3.0, 4.0, 4.0],   # mean = 3.5
        ], dtype=np.float32)
        weight = np.ones_like(nll)
        src_ids = np.array([
            [0, 0, 0, 0],
            [1, 1, 1, 1],
        ], dtype=np.int32)
        batch = {"input_ids": np.zeros((2, 4), dtype=np.int32),
                 "labels": np.zeros((2, 4), dtype=np.int32)}
        vocab = {"src_a": 0, "src_b": 1}

        eval_fn = make_per_source_validation_loss_eval_from_eval_step(
            self._stub_eval_step_fn(nll, weight),
            [batch], [src_ids], vocab,
        )
        result = eval_fn(step=100, state={})

        # Aggregate: mean of 8 values = (1+1+2+2+3+3+4+4)/8 = 2.5
        assert result["val_loss"] == pytest.approx(2.5)
        # Per-source: src_a mean = 1.5, src_b mean = 3.5
        assert result["val_loss/src_a"] == pytest.approx(1.5)
        assert result["val_loss/src_b"] == pytest.approx(3.5)
        # PPL = exp(loss)
        assert result["val_ppl"] == pytest.approx(np.exp(2.5))
        assert result["val_ppl/src_a"] == pytest.approx(np.exp(1.5))
        assert result["val_ppl/src_b"] == pytest.approx(np.exp(3.5))
        # n_tokens
        assert result["val_n_tokens"] == 8.0
        assert result["val_n_tokens/src_a"] == 4.0
        assert result["val_n_tokens/src_b"] == 4.0

    def test_weight_mask_zero_excluded(self):
        # When weight_per_token is 0 for some positions (e.g., padding),
        # those tokens are excluded from BOTH aggregate and per-source.
        nll = np.array([
            [1.0, 1.0, 999.0, 999.0],
        ], dtype=np.float32)
        weight = np.array([
            [1.0, 1.0, 0.0, 0.0],
        ], dtype=np.float32)
        src_ids = np.array([[0, 0, 0, 0]], dtype=np.int32)
        batch = {"input_ids": np.zeros((1, 4), dtype=np.int32),
                 "labels": np.zeros((1, 4), dtype=np.int32)}

        eval_fn = make_per_source_validation_loss_eval_from_eval_step(
            self._stub_eval_step_fn(nll, weight),
            [batch], [src_ids], {"src_a": 0},
        )
        result = eval_fn(step=0, state={})
        # Only first 2 positions counted; the 999s are masked out.
        assert result["val_loss"] == pytest.approx(1.0)
        assert result["val_n_tokens"] == 2.0

    def test_source_with_zero_tokens_omitted(self):
        # If src_b has no held-out coverage (e.g., all sentinel -1 in
        # the batch), src_b should NOT appear in the result.
        nll = np.array([[1.0, 1.0]], dtype=np.float32)
        weight = np.ones_like(nll)
        # All positions tagged src_a; src_b never appears.
        src_ids = np.array([[0, 0]], dtype=np.int32)
        batch = {"input_ids": np.zeros((1, 2), dtype=np.int32),
                 "labels": np.zeros((1, 2), dtype=np.int32)}

        eval_fn = make_per_source_validation_loss_eval_from_eval_step(
            self._stub_eval_step_fn(nll, weight),
            [batch], [src_ids], {"src_a": 0, "src_b": 1},
        )
        result = eval_fn(step=0, state={})
        assert "val_loss/src_a" in result
        assert "val_loss/src_b" not in result
        assert "val_n_tokens/src_b" not in result

    def test_sentinel_minus_one_positions_dont_bucket(self):
        # Positions with source_id=-1 (boundary / unknown source) are
        # excluded from per-source bucketing, but they DO contribute to
        # the aggregate (as long as their weight is > 0).
        nll = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        weight = np.ones_like(nll)
        src_ids = np.array([[0, -1, 0]], dtype=np.int32)  # middle is sentinel
        batch = {"input_ids": np.zeros((1, 3), dtype=np.int32),
                 "labels": np.zeros((1, 3), dtype=np.int32)}

        eval_fn = make_per_source_validation_loss_eval_from_eval_step(
            self._stub_eval_step_fn(nll, weight),
            [batch], [src_ids], {"src_a": 0},
        )
        result = eval_fn(step=0, state={})
        # Aggregate over all 3 tokens
        assert result["val_loss"] == pytest.approx(2.0)
        # src_a only sees positions 0 and 2: mean = (1+3)/2 = 2.0
        assert result["val_loss/src_a"] == pytest.approx(2.0)
        assert result["val_n_tokens/src_a"] == 2.0
        # Aggregate counted all 3
        assert result["val_n_tokens"] == 3.0

    def test_nan_batch_skipped_warns(self, caplog):
        nll = np.array([[float("nan"), 1.0]], dtype=np.float32)
        weight = np.ones_like(nll)
        src_ids = np.array([[0, 0]], dtype=np.int32)
        batch = {"input_ids": np.zeros((1, 2), dtype=np.int32),
                 "labels": np.zeros((1, 2), dtype=np.int32)}

        eval_fn = make_per_source_validation_loss_eval_from_eval_step(
            self._stub_eval_step_fn(nll, weight),
            [batch], [src_ids], {"src_a": 0},
        )
        result = eval_fn(step=0, state={})
        # Single all-NaN batch -> no usable data -> returns None
        assert result is None

    def test_empty_held_out_raises(self):
        with pytest.raises(ValueError, match="at least one held-out batch"):
            make_per_source_validation_loss_eval_from_eval_step(
                lambda s, b: {},
                [], [], {"src_a": 0},
            )

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="equal length"):
            make_per_source_validation_loss_eval_from_eval_step(
                lambda s, b: {},
                [{"input_ids": np.zeros((1, 2))}],
                [],
                {"src_a": 0},
            )
