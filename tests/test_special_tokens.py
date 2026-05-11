"""Tests for the special-tokens constants module."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from myllm.data.special_tokens import (
    OPTIONAL,
    REQUIRED,
    SpecialTokens,
    all_special_token_strings,
    get_required_ids,
    verify_tokenizer_has_required,
)


class TestConstants:
    def test_required_set_is_disjoint_from_optional(self):
        assert set(REQUIRED).isdisjoint(set(OPTIONAL))

    def test_required_contains_eos_pad_bos(self):
        assert SpecialTokens.EOS in REQUIRED
        assert SpecialTokens.PAD in REQUIRED
        assert SpecialTokens.BOS in REQUIRED

    def test_string_forms_are_unique(self):
        all_strs = list(REQUIRED) + list(OPTIONAL)
        assert len(set(all_strs)) == len(all_strs)


class TestAllSpecialTokenStrings:
    def test_no_reserved_slots(self):
        out = all_special_token_strings(reserved_slots=0)
        assert out == list(REQUIRED) + list(OPTIONAL)

    def test_with_reserved_slots(self):
        out = all_special_token_strings(reserved_slots=4)
        assert len(out) == len(REQUIRED) + len(OPTIONAL) + 4
        for i in range(4):
            assert f"<|extra_{i}|>" in out

    def test_deterministic_ordering(self):
        a = all_special_token_strings(reserved_slots=8)
        b = all_special_token_strings(reserved_slots=8)
        assert a == b


class TestVerifyTokenizerHasRequired:
    def _make_tokenizer(self, present: set[str]):
        tok = MagicMock()
        tok.token_to_id.side_effect = lambda s: 1 if s in present else None
        return tok

    def test_complete_tokenizer_passes(self):
        tok = self._make_tokenizer(set(REQUIRED) | set(OPTIONAL))
        verify_tokenizer_has_required(tok)  # no raise

    def test_missing_required_raises(self):
        tok = self._make_tokenizer(set(REQUIRED) - {SpecialTokens.EOS})
        with pytest.raises(ValueError) as exc:
            verify_tokenizer_has_required(tok)
        assert SpecialTokens.EOS in str(exc.value)

    def test_missing_optional_does_not_raise(self):
        # Only required tokens are present — should still pass.
        tok = self._make_tokenizer(set(REQUIRED))
        verify_tokenizer_has_required(tok)


class TestGetRequiredIds:
    def test_returns_id_map(self):
        tok = MagicMock()
        tok.token_to_id.side_effect = lambda s: hash(s) % 100000
        ids = get_required_ids(tok)
        assert set(ids.keys()) == set(REQUIRED)
        for v in ids.values():
            assert isinstance(v, int)

    def test_raises_if_missing(self):
        tok = MagicMock()
        tok.token_to_id.return_value = None
        with pytest.raises(ValueError):
            get_required_ids(tok)
