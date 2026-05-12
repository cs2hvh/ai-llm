"""Tests for the canary ladder infrastructure (src/myllm/canary.py).

L0 + L5 unit tests cover the functions directly. L3 has its own
script (scripts/canary_l3_resume.py) and a smoke test below — full
end-to-end is slow (spawns three subprocess training runs) so we
gate it behind a marker.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from myllm.canary import (
    CheckResult,
    StageResult,
    hash_training_state,
    l0_check_model_config_self_consistency,
    l0_check_tokenizer_roundtrip,
    l5_check_corpus_manifest_complete,
    l5_check_segment_ids_well_formed,
    l5_check_source_share_drift,
    l5_check_token_range,
    l5_check_tokenizer_sha_uniform,
    report_to_json,
    report_to_text,
)
from myllm.data.packed_corpus import (
    DocSpan,
    PackedCorpusWriter,
    TOKEN_DTYPE,
    write_corpus_manifest,
)


# --------------------------------------------------------------------------- #
# CheckResult / StageResult shape
# --------------------------------------------------------------------------- #
class TestResultShapes:
    def test_check_result_to_dict(self):
        r = CheckResult(name="x", passed=True, summary="ok")
        assert r.to_dict()["name"] == "x"

    def test_stage_passed_aggregates(self):
        s = StageResult(stage="L0", checks=[
            CheckResult("a", True, "ok"),
            CheckResult("b", True, "ok"),
        ])
        assert s.passed is True
        s2 = StageResult(stage="L0", checks=[
            CheckResult("a", True, "ok"),
            CheckResult("b", False, "fail"),
        ])
        assert s2.passed is False

    def test_report_to_json_round_trips(self):
        stages = [StageResult(stage="L0", checks=[CheckResult("a", True, "ok")])]
        out = json.loads(report_to_json(stages))
        assert out[0]["stage"] == "L0"
        assert out[0]["passed"] is True

    def test_report_to_text_indicates_overall_status(self):
        passing = [StageResult(stage="L0", checks=[CheckResult("a", True, "ok")])]
        assert "PASS" in report_to_text(passing)
        failing = [StageResult(stage="L0", checks=[CheckResult("a", False, "x")])]
        assert "FAIL" in report_to_text(failing)


# --------------------------------------------------------------------------- #
# L0 — Model config self-consistency
# --------------------------------------------------------------------------- #
class TestL0ConfigSelfConsistency:
    def _write_cfg(self, tmp_path, overrides: dict) -> Path:
        base = {
            "layers": 16,
            "hidden_dim": 2048,
            "ffn_dim": 8192,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 64,
            "vocab_size": 131072,
            "context_length": 8192,
            "rope_base": 500000.0,
        }
        base.update(overrides)
        import yaml
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.safe_dump(base))
        return path

    def test_valid_config_passes(self, tmp_path):
        path = self._write_cfg(tmp_path, {})
        r = l0_check_model_config_self_consistency(path)
        assert r.passed
        assert r.details["hidden"] == 2048
        assert r.details["ffn_ratio"] == 4.0

    def test_hidden_not_divisible_by_heads_fails(self, tmp_path):
        # 2049 hidden / 32 heads → not divisible
        path = self._write_cfg(tmp_path, {"hidden_dim": 2049, "ffn_dim": 8196})
        r = l0_check_model_config_self_consistency(path)
        assert not r.passed
        assert "hidden_dim" in r.summary

    def test_kv_heads_not_divisor_of_num_heads_fails(self, tmp_path):
        # 32 heads / 5 KV → ill-defined GQA
        path = self._write_cfg(tmp_path, {"num_kv_heads": 5})
        r = l0_check_model_config_self_consistency(path)
        assert not r.passed
        assert "num_kv_heads" in r.summary

    def test_negative_rope_base_fails(self, tmp_path):
        path = self._write_cfg(tmp_path, {"rope_base": -1.0})
        r = l0_check_model_config_self_consistency(path)
        assert not r.passed
        assert "rope_base" in r.summary

    def test_ffn_ratio_outside_band_fails(self, tmp_path):
        # 2048 hidden + 100 ffn → ratio ~0.05 → below sane band
        path = self._write_cfg(tmp_path, {"ffn_dim": 100})
        r = l0_check_model_config_self_consistency(path)
        assert not r.passed
        assert "ffn" in r.summary

    def test_missing_required_field_fails(self, tmp_path):
        import yaml
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.safe_dump({"layers": 1}))  # missing most fields
        r = l0_check_model_config_self_consistency(p)
        assert not r.passed
        assert "missing" in r.summary

    def test_live_base_1b_config_passes(self):
        """The repo's actual configs/base_1b.yaml must pass L0."""
        path = Path(__file__).resolve().parents[1] / "configs" / "base_1b.yaml"
        r = l0_check_model_config_self_consistency(path)
        assert r.passed, f"base_1b.yaml fails L0: {r.summary}"


# --------------------------------------------------------------------------- #
# L0 — Tokenizer round-trip
# --------------------------------------------------------------------------- #
class TestL0TokenizerRoundtrip:
    def test_missing_tokenizer_file_fails(self, tmp_path):
        r = l0_check_tokenizer_roundtrip(tmp_path / "nope.json")
        assert not r.passed
        assert "not found" in r.summary


# --------------------------------------------------------------------------- #
# L5 — Packed corpus sanity (use a synthetic tiny corpus)
# --------------------------------------------------------------------------- #
def _make_tiny_corpus(
    tmp_path: Path,
    *,
    n_sequences: int = 4,
    sequence_length: int = 8,
    tokenizer_sha256: str = "deadbeef",
    target_share: dict | None = None,
    actual_share_override: dict | None = None,
) -> Path:
    root = tmp_path / "c"
    w = PackedCorpusWriter(
        root,
        sequence_length=sequence_length,
        sequences_per_shard=2,
        tokenizer_sha256=tokenizer_sha256,
    )
    for sid in range(n_sequences):
        # Token pattern designed to span [0, 200) so vocab_size=100 actually
        # finds out-of-range tokens. The previous (sid*1000+i) % 200 collapsed
        # to [0..7] because 1000 % 200 = 0.
        tokens = np.array(
            [((sid * 37 + i * 23 + 13) % 200) for i in range(sequence_length)],
            dtype=TOKEN_DTYPE,
        )
        spans = [DocSpan(
            -1, -1, "a" if sid % 2 == 0 else "b",
            sid, "rev-1", 0, sequence_length, sid,
        )]
        w.append_sequence(tokens, spans)
    w.close()
    write_corpus_manifest(
        root,
        corpus_name="c",
        tokenizer_sha256=tokenizer_sha256,
        sequence_length=sequence_length,
        sequences_per_shard=2,
        source_revisions={"a": "rev-1", "b": "rev-1"},
        target_source_share=(target_share or {"a": 0.5, "b": 0.5}),
    )
    if actual_share_override is not None:
        mp = root / "manifest.json"
        d = json.loads(mp.read_text())
        d["actual_source_share"] = actual_share_override
        mp.write_text(json.dumps(d))
    return root


class TestL5ManifestComplete:
    def test_complete_corpus_passes(self, tmp_path):
        root = _make_tiny_corpus(tmp_path)
        r = l5_check_corpus_manifest_complete(root)
        assert r.passed

    def test_missing_shard_manifest_fails(self, tmp_path):
        root = _make_tiny_corpus(tmp_path)
        # Remove a shard manifest to simulate partial write.
        (root / "shard-000000" / "manifest.json").unlink()
        r = l5_check_corpus_manifest_complete(root)
        assert not r.passed
        assert "lack manifest" in r.summary


class TestL5TokenizerShaUniform:
    def test_uniform_tokenizer_passes(self, tmp_path):
        root = _make_tiny_corpus(tmp_path)
        r = l5_check_tokenizer_sha_uniform(root)
        assert r.passed

    def test_tampered_shard_sha_fails(self, tmp_path):
        root = _make_tiny_corpus(tmp_path)
        # Tamper with shard-000000's tokenizer_sha256.
        mp = root / "shard-000000" / "manifest.json"
        d = json.loads(mp.read_text())
        d["tokenizer_sha256"] = "WRONG_SHA"
        mp.write_text(json.dumps(d))
        r = l5_check_tokenizer_sha_uniform(root)
        assert not r.passed
        assert "differing tokenizer" in r.summary


class TestL5SourceShareDrift:
    def test_no_drift_passes(self, tmp_path):
        # _make_tiny_corpus produces a 50/50 mix matching target 50/50.
        root = _make_tiny_corpus(tmp_path)
        r = l5_check_source_share_drift(root, tolerance=0.01)
        assert r.passed

    def test_big_drift_fails(self, tmp_path):
        # Override actual share to be 80/20 — way off from 50/50 target.
        root = _make_tiny_corpus(
            tmp_path, actual_share_override={"a": 0.8, "b": 0.2},
        )
        r = l5_check_source_share_drift(root, tolerance=0.05)
        assert not r.passed
        assert "exceeds tolerance" in r.summary


class TestL5TokenRange:
    def test_all_tokens_in_range_passes(self, tmp_path):
        root = _make_tiny_corpus(tmp_path)
        # Tokens are < 200; vocab_size 256 should pass.
        r = l5_check_token_range(root, vocab_size=256, n_samples=4)
        assert r.passed

    def test_oversized_token_fails(self, tmp_path):
        root = _make_tiny_corpus(tmp_path)
        # vocab_size=100 means tokens >= 100 are out of range.
        r = l5_check_token_range(root, vocab_size=100, n_samples=4)
        assert not r.passed
        assert "vocab_size" in r.summary


class TestL5SegmentIds:
    def test_well_formed_segments_pass(self, tmp_path):
        root = _make_tiny_corpus(tmp_path)
        r = l5_check_segment_ids_well_formed(root, n_samples=4)
        assert r.passed


# --------------------------------------------------------------------------- #
# State hash determinism
# --------------------------------------------------------------------------- #
class TestHashTrainingState:
    def test_same_state_same_hash(self):
        s = {"step": 5, "data_position": 1024,
             "trainable_variables": {"w": np.array([1.0, 2.0, 3.0])}}
        h1 = hash_training_state(s)
        h2 = hash_training_state(s)
        assert h1 == h2

    def test_differing_state_differing_hash(self):
        s1 = {"step": 5, "data_position": 1024}
        s2 = {"step": 5, "data_position": 1025}
        assert hash_training_state(s1) != hash_training_state(s2)

    def test_key_order_doesnt_matter(self):
        s1 = {"step": 5, "data_position": 1024}
        s2 = {"data_position": 1024, "step": 5}
        assert hash_training_state(s1) == hash_training_state(s2)
