"""Tests for the teacher top-K mass audit (scripts/audit_teacher_topk_mass.py).

The audit is GPU-blocked end-to-end (loading a 32B-param teacher is
infeasible on CPU), but the math itself is pure numpy and validates
the K=8 → K=16 escalation logic without GPU.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))

from audit_teacher_topk_mass import (  # noqa: E402
    recommend_k,
    run_audit,
    softmax_topk_mass,
    summarize_masses,
)


# --------------------------------------------------------------------------- #
# softmax_topk_mass
# --------------------------------------------------------------------------- #
class TestSoftmaxTopKMass:
    def test_uniform_logits_top_k_mass_equals_k_over_v(self):
        # Uniform logits → uniform softmax. Top-K mass should be exactly K/V.
        n, v = 4, 16
        logits = np.zeros((n, v), dtype=np.float32)
        result = softmax_topk_mass(logits, ks=[4, 8])
        np.testing.assert_allclose(result[4], np.full(n, 4 / 16), rtol=1e-6)
        np.testing.assert_allclose(result[8], np.full(n, 8 / 16), rtol=1e-6)

    def test_one_hot_logits_top_1_mass_near_1(self):
        # Sharply peaked: one logit at 100, others at 0 → top-1 mass ≈ 1.
        n, v = 3, 32
        logits = np.zeros((n, v), dtype=np.float32)
        logits[:, 7] = 100.0  # arg-max at index 7 for every row
        result = softmax_topk_mass(logits, ks=[1, 8])
        assert (result[1] > 0.999).all(), f"top-1: {result[1]}"
        assert (result[8] >= result[1] - 1e-6).all()

    def test_monotonic_in_k(self):
        # top-K mass must be non-decreasing in K, by construction.
        rng = np.random.default_rng(0)
        logits = rng.normal(size=(20, 64)).astype(np.float32)
        result = softmax_topk_mass(logits, ks=[2, 4, 8, 16, 32])
        prev = np.zeros(20, dtype=np.float32)
        for k in [2, 4, 8, 16, 32]:
            assert (result[k] >= prev - 1e-6).all(), f"K={k} not >= K={k//2}"
            prev = result[k]

    def test_full_vocab_mass_is_one(self):
        rng = np.random.default_rng(1)
        n, v = 5, 50
        logits = rng.normal(size=(n, v)).astype(np.float32)
        result = softmax_topk_mass(logits, ks=[v])
        np.testing.assert_allclose(result[v], np.ones(n), atol=1e-5)

    def test_matches_explicit_softmax(self):
        # Cross-check against an O(NV log V) explicit-sort reference.
        rng = np.random.default_rng(2)
        logits = rng.normal(size=(10, 30)).astype(np.float32) * 2.0
        result = softmax_topk_mass(logits, ks=[3, 7])
        # Reference: full softmax, sort descending, sum first K.
        shifted = logits - logits.max(axis=1, keepdims=True)
        sm = np.exp(shifted)
        sm /= sm.sum(axis=1, keepdims=True)
        for k in (3, 7):
            ref = np.sort(sm, axis=1)[:, ::-1][:, :k].sum(axis=1)
            np.testing.assert_allclose(result[k], ref.astype(np.float32), atol=1e-5)

    def test_rejects_empty_ks(self):
        with pytest.raises(ValueError, match="non-empty"):
            softmax_topk_mass(np.zeros((1, 4), dtype=np.float32), ks=[])

    def test_rejects_k_gt_vocab(self):
        with pytest.raises(ValueError, match="exceeds vocab"):
            softmax_topk_mass(np.zeros((1, 4), dtype=np.float32), ks=[10])


# --------------------------------------------------------------------------- #
# summarize_masses
# --------------------------------------------------------------------------- #
class TestSummarizeMasses:
    def test_basic_quantiles(self):
        x = np.linspace(0, 1, 101, dtype=np.float32)
        out = summarize_masses(x, thresholds=(0.5, 0.9))
        assert abs(out["p50"] - 0.5) < 1e-3
        assert abs(out["p10"] - 0.1) < 1e-3
        assert abs(out["p90"] - 0.9) < 1e-3
        assert abs(out["mean"] - 0.5) < 1e-3
        # Fraction below 0.5: 50/101 ≈ 0.495 (strict less-than)
        assert 0.48 < out["frac_below_0.50"] < 0.51
        assert 0.88 < out["frac_below_0.90"] < 0.91

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            summarize_masses(np.array([], dtype=np.float32))


# --------------------------------------------------------------------------- #
# recommend_k
# --------------------------------------------------------------------------- #
class TestRecommendK:
    def test_picks_smallest_safe_k(self):
        # K=4 fails the threshold (15% below 0.95); K=8 passes (5% below).
        by_k = {
            4: {"frac_below_0.95": 0.15},
            8: {"frac_below_0.95": 0.05},
            16: {"frac_below_0.95": 0.01},
        }
        d = recommend_k(by_k, target_mass=0.95, max_frac_below=0.10)
        assert d["recommended_k"] == 8
        assert "K=8" in d["rationale"]

    def test_escalates_when_no_k_safe(self):
        # All Ks fail; recommendation = largest available with a clear caveat.
        by_k = {
            4: {"frac_below_0.95": 0.50},
            8: {"frac_below_0.95": 0.30},
            16: {"frac_below_0.95": 0.20},
        }
        d = recommend_k(by_k, target_mass=0.95, max_frac_below=0.10)
        assert d["recommended_k"] == 16
        assert "consider K > 16" in d["rationale"]

    def test_accepts_string_keys(self):
        # JSON-loaded dicts use string keys.
        by_k = {"4": {"frac_below_0.95": 0.15}, "8": {"frac_below_0.95": 0.05}}
        d = recommend_k(by_k)
        assert d["recommended_k"] == 8


# --------------------------------------------------------------------------- #
# End-to-end with synthetic teacher + small corpus
# --------------------------------------------------------------------------- #
class TestRunAuditSyntheticEndToEnd:
    def _make_corpus(self, tmp_path: Path, n_tokens: int = 1024) -> Path:
        # uint32 token ids — the same format _iter_tokenized_corpus expects.
        arr = np.arange(n_tokens, dtype=np.uint32)
        p = tmp_path / "audit_corpus.bin"
        arr.tofile(p)
        return p

    def test_e2e_synthetic_returns_well_formed_summary(self, tmp_path: Path):
        corpus = self._make_corpus(tmp_path, n_tokens=512)
        out = run_audit(
            teacher_id="synth-test",
            teacher_hf_model="synth-model",
            corpus_path=corpus,
            n_positions=64,
            ks=[4, 8, 16, 32],
            batch_size=2,
            sequence_length=32,
            synthetic=True,
        )
        # Top-level shape:
        assert out["teacher"] == "synth-test"
        assert out["synthetic"] is True
        assert out["n_positions"] == 64
        assert sorted(out["by_k"].keys()) == ["16", "32", "4", "8"]
        # Each K should have the standard summary fields:
        for k in ("4", "8", "16", "32"):
            stats = out["by_k"][k]
            for field in ("mean", "p10", "p50", "p90", "p99",
                          "frac_below_0.90", "frac_below_0.95", "frac_below_0.99"):
                assert field in stats, f"K={k} missing {field}"
                assert 0.0 <= stats[field] <= 1.0
        # decision block:
        assert "recommended_k" in out["decision"]
        assert out["decision"]["recommended_k"] in (4, 8, 16, 32)

    def test_synthetic_topk_mass_is_tiny_for_v131k(self, tmp_path: Path):
        # Synthetic logits are gaussian N(0,1) over V=131072. Top-8 mass
        # of a uniform-ish soft distribution should be very small — this
        # is the "code/math tail" stress-test scenario, where K=8 captures
        # almost no mass.
        corpus = self._make_corpus(tmp_path, n_tokens=512)
        out = run_audit(
            teacher_id="synth-flat",
            teacher_hf_model="some-flat-teacher",
            corpus_path=corpus,
            n_positions=64,
            ks=[8],
            batch_size=2,
            sequence_length=32,
            synthetic=True,
        )
        # Expect top-8 mean mass < 0.1 — the synthetic teacher is broad.
        assert out["by_k"]["8"]["mean"] < 0.1
        # And the recommended K should escalate since no K satisfies the
        # 0.95-mass / 10% threshold among the candidate Ks.
        assert out["decision"]["recommended_k"] == 8  # only K=8 was tested
