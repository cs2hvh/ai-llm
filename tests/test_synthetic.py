"""Tests for the synthetic-data iterator."""
from __future__ import annotations

import pytest

from myllm.data import make_synthetic_data_iter


class TestSyntheticDataIter:
    def test_shape_and_dtype(self):
        it = make_synthetic_data_iter(
            micro_batch=4, sequence_length=16, vocab_size=100, n_steps=3
        )
        for b in it:
            assert b["input_ids"].shape == (4, 16)
            assert b["labels"].shape == (4, 16)
            assert b["input_ids"].dtype.name == "int32"

    def test_input_label_shifted_by_one(self):
        # Same seed → same draw → input[t+1] == label[t].
        it = make_synthetic_data_iter(
            micro_batch=2, sequence_length=8, vocab_size=50, n_steps=1, seed=7
        )
        b = next(iter(it))
        # We sampled S+1 tokens then split: input is first S, label is last S.
        # So input[:, 1:] == label[:, :-1].
        assert (b["input_ids"][:, 1:] == b["labels"][:, :-1]).all()

    def test_n_steps_cap(self):
        it = make_synthetic_data_iter(
            micro_batch=1, sequence_length=4, vocab_size=10, n_steps=5
        )
        assert sum(1 for _ in it) == 5

    def test_infinite_when_n_steps_none(self):
        it = make_synthetic_data_iter(
            micro_batch=1, sequence_length=4, vocab_size=10, n_steps=None
        )
        # Pull more than any reasonable cap.
        for _ in range(100):
            next(it)

    def test_reproducible_with_same_seed(self):
        a = next(iter(make_synthetic_data_iter(2, 8, 50, n_steps=1, seed=42)))
        b = next(iter(make_synthetic_data_iter(2, 8, 50, n_steps=1, seed=42)))
        assert (a["input_ids"] == b["input_ids"]).all()
        assert (a["labels"] == b["labels"]).all()

    def test_invalid_args(self):
        with pytest.raises(ValueError):
            next(iter(make_synthetic_data_iter(0, 8, 50)))
        with pytest.raises(ValueError):
            next(iter(make_synthetic_data_iter(2, 1, 50)))
        with pytest.raises(ValueError):
            next(iter(make_synthetic_data_iter(2, 8, 1)))
