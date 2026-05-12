"""Tests for the governance card renderer.

These guard against:
  - Template drift (renderer + on-disk get out of sync silently).
  - AUTO-block substitution bugs (wrong block name, malformed markers).
  - Format regressions (e.g. vocab 131072 reading as "128k" again).
  - Sources block missing rows when the config grows.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

# Import the script as a module (it lives in scripts/, not a package).
_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "render_governance_cards.py"
_spec = importlib.util.spec_from_file_location("render_governance_cards", _SCRIPT)
rgc = importlib.util.module_from_spec(_spec)
sys.modules["render_governance_cards"] = rgc
_spec.loader.exec_module(rgc)


# --------------------------------------------------------------------------- #
# Architecture renderer
# --------------------------------------------------------------------------- #
class TestArchitectureBlock:
    def _base_model_cfg(self) -> dict:
        return {
            "layers": 16,
            "hidden_dim": 2048,
            "ffn_dim": 8192,
            "num_heads": 32,
            "num_kv_heads": 8,
            "vocab_size": 131072,
            "tie_embeddings": True,
            "context_length": 8192,
            "context_extension_yarn_target": 32768,
            "rope_base": 500000.0,
            "norm": "rmsnorm",
            "activation": "swiglu",
            "qk_norm": True,
            "mixed_precision": "bfloat16",
        }

    def test_renders_known_architecture_line(self):
        out = rgc.render_architecture_block(self._base_model_cfg(), None)
        assert "16 layers × 2048 hidden" in out
        assert "4× FFN" in out
        assert "GQA 32:8" in out
        assert "tied embeddings" in out
        assert "QK-norm" in out

    def test_vocab_size_131072_renders_as_131k(self):
        """Regression: integer truncation made 131072 → 128k."""
        out = rgc.render_architecture_block(self._base_model_cfg(), None)
        assert "131k vocab" in out
        assert "128k vocab" not in out

    def test_token_budget_from_base_cfg_when_available(self):
        out = rgc.render_architecture_block(
            self._base_model_cfg(),
            {"target_tokens_v1": 1_000_000_000_000},
        )
        assert "1T" in out

    def test_token_budget_falls_back_to_model_target(self):
        """When no base_cfg (e.g. rendering a wind-tunnel card), the model
        config's own target_tokens is used."""
        cfg = self._base_model_cfg()
        cfg["target_tokens"] = 500_000_000
        out = rgc.render_architecture_block(cfg, None)
        assert "500M" in out

    def test_yarn_extension_only_appears_when_configured(self):
        cfg = self._base_model_cfg()
        del cfg["context_extension_yarn_target"]
        out = rgc.render_architecture_block(cfg, None)
        assert "YaRN" not in out

    def test_no_qk_norm_line_when_disabled(self):
        cfg = self._base_model_cfg()
        cfg["qk_norm"] = False
        out = rgc.render_architecture_block(cfg, None)
        assert "QK-norm" not in out


# --------------------------------------------------------------------------- #
# Sources renderer
# --------------------------------------------------------------------------- #
class TestSourcesBlock:
    def test_renders_share_as_percent(self):
        cfg = {
            "sources": [
                {"dataset": "HuggingFaceFW/fineweb-edu", "share": 0.44},
                {"dataset": "pg19", "share": 0.05},
            ]
        }
        out = rgc.render_sources_block(cfg)
        assert "44.0%" in out
        assert "5.0%" in out

    def test_split_appears_in_dataset_label(self):
        cfg = {
            "sources": [
                {
                    "dataset": "ai4bharat/sangraha",
                    "share": 0.04,
                    "config_name": "verified",
                    "split": "hin",
                }
            ]
        }
        out = rgc.render_sources_block(cfg)
        assert "split=hin" in out
        assert "verified" in out

    def test_total_share_summed(self):
        cfg = {
            "sources": [
                {"dataset": "a", "share": 0.4},
                {"dataset": "b", "share": 0.6},
            ]
        }
        out = rgc.render_sources_block(cfg)
        assert "Total**: 100.0%" in out

    def test_license_lookup_for_known_dataset(self):
        cfg = {"sources": [{"dataset": "HuggingFaceFW/fineweb-edu", "share": 0.44}]}
        out = rgc.render_sources_block(cfg)
        assert "ODC-By" in out

    def test_unknown_dataset_falls_back_to_register_pointer(self):
        cfg = {"sources": [{"dataset": "some/new-dataset", "share": 0.1}]}
        out = rgc.render_sources_block(cfg)
        assert "see license_register.md" in out

    def test_row_count_matches_config(self):
        cfg = {"sources": [{"dataset": f"ds{i}", "share": 0.1} for i in range(5)]}
        out = rgc.render_sources_block(cfg)
        # Each row is one table line; count by occurrences of `| ds`
        assert out.count("| ds") == 5


# --------------------------------------------------------------------------- #
# Template substitution
# --------------------------------------------------------------------------- #
class TestRenderTemplate:
    def test_substitutes_architecture_block(self):
        template = (
            "before\n"
            '<!-- AUTO:start name="architecture" -->\n'
            "OLD CONTENT\n"
            "<!-- AUTO:end -->\n"
            "after\n"
        )
        configs = {
            "model": {
                "layers": 4,
                "hidden_dim": 256,
                "ffn_dim": 1024,
                "num_heads": 4,
                "num_kv_heads": 4,
                "vocab_size": 32000,
                "context_length": 1024,
                "rope_base": 10000.0,
                "norm": "rmsnorm",
                "activation": "swiglu",
                "qk_norm": False,
            },
            "base": None,
        }
        out = rgc.render_template(template, configs=configs)
        assert "OLD CONTENT" not in out
        assert "4 layers × 256 hidden" in out
        assert "before" in out and "after" in out

    def test_substitutes_sources_block(self):
        template = (
            '<!-- AUTO:start name="sources" -->\nx\n<!-- AUTO:end -->'
        )
        configs = {
            "model": {},
            "data": {"sources": [{"dataset": "foo", "share": 1.0}]},
        }
        out = rgc.render_template(template, configs=configs)
        assert "| foo |" in out

    def test_unknown_auto_block_name_raises(self):
        template = (
            '<!-- AUTO:start name="not-a-real-block" -->\nx\n<!-- AUTO:end -->'
        )
        with pytest.raises(ValueError, match="unknown AUTO block name"):
            rgc.render_template(template, configs={"model": {}, "data": {}})

    def test_idempotent(self):
        """Running the renderer twice produces byte-identical output."""
        template = (
            '<!-- AUTO:start name="architecture" -->\n'
            "stale\n"
            "<!-- AUTO:end -->"
        )
        configs = {
            "model": {
                "layers": 1,
                "hidden_dim": 64,
                "ffn_dim": 256,
                "num_heads": 1,
                "num_kv_heads": 1,
                "vocab_size": 1024,
                "context_length": 512,
                "rope_base": 10000.0,
            },
            "base": None,
        }
        first = rgc.render_template(template, configs=configs)
        second = rgc.render_template(first, configs=configs)
        assert first == second

    def test_preserves_static_content_outside_auto_markers(self):
        template = (
            "# Header\n\n"
            "Static prose that must not be touched.\n\n"
            '<!-- AUTO:start name="architecture" -->\n'
            "old\n"
            "<!-- AUTO:end -->\n\n"
            "More static prose.\n"
        )
        configs = {
            "model": {
                "layers": 1,
                "hidden_dim": 64,
                "ffn_dim": 256,
                "num_heads": 1,
                "num_kv_heads": 1,
                "vocab_size": 1024,
                "context_length": 512,
                "rope_base": 10000.0,
            },
            "base": None,
        }
        out = rgc.render_template(template, configs=configs)
        assert "Static prose that must not be touched." in out
        assert "More static prose." in out


# --------------------------------------------------------------------------- #
# Real-config smoke test — guarantee the live configs render cleanly.
# This is the most valuable test: if someone breaks base_1b.yaml or
# pretrain_mix.yaml in a way the renderer can't handle, CI fails here
# instead of at release time.
# --------------------------------------------------------------------------- #
class TestLiveConfigs:
    REPO = Path(__file__).resolve().parents[1]

    def test_base_1b_renders_without_error(self):
        with open(self.REPO / "configs" / "base_1b.yaml") as f:
            model_cfg = yaml.safe_load(f)
        out = rgc.render_architecture_block(model_cfg, model_cfg)
        assert "layers" in out
        assert "Architecture" in out

    def test_pretrain_mix_renders_without_error(self):
        with open(self.REPO / "configs" / "data" / "pretrain_mix.yaml") as f:
            data_cfg = yaml.safe_load(f)
        out = rgc.render_sources_block(data_cfg)
        assert "Total" in out
        # Sanity: the live config sums to 100% exactly.
        assert "100.0%" in out
