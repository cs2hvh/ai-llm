from __future__ import annotations

from pathlib import Path

import pytest

from myllm_pre2.config import load_data_mix_config, load_dense_config


REPO = Path(__file__).resolve().parents[1]


def test_pre2_dense_1_5b_param_estimate_matches_plan():
    cfg = load_dense_config(REPO / "configs" / "pre2_dense_1_5b.yaml")

    assert cfg.name == "myllm-pre2-dense-1.5b-base"
    assert cfg.model.layers == 20
    assert cfg.model.hidden_dim == 2048
    assert cfg.model.attention.num_heads == 32
    assert cfg.model.attention.num_kv_heads == 8
    assert cfg.model.embeddings.tied is True
    assert cfg.parameter_count_estimate() == 1_484_874_240
    assert cfg.parameter_target_delta() < 0.01


def test_pre2_dense_1_5b_token_budget_is_mainline_decision():
    cfg = load_dense_config(REPO / "configs" / "pre2_dense_1_5b.yaml")

    assert cfg.training.token_budget.internal_minimum == 1_000_000_000_000
    assert cfg.training.token_budget.release_target == 1_500_000_000_000
    assert cfg.training.token_budget.stretch == 3_000_000_000_000
    assert cfg.training_steps_for_tokens(cfg.training.token_budget.release_target) == 715_256
    assert cfg.dense_training_flops(
        cfg.training.token_budget.release_target
    ) == 6 * cfg.parameter_count_estimate() * cfg.training.token_budget.release_target


def test_pre2_data_stage_tokens_match_release_target():
    cfg = load_dense_config(REPO / "configs" / "pre2_dense_1_5b.yaml")
    stage_paths = [
        REPO / "configs" / "data" / "pre2_mix_stage1.yaml",
        REPO / "configs" / "data" / "pre2_mix_stage2.yaml",
        REPO / "configs" / "data" / "pre2_mix_anneal.yaml",
    ]
    total = sum(load_data_mix_config(path).target_tokens for path in stage_paths)

    assert total == cfg.training.token_budget.release_target


def test_pre2_dense_mainline_disables_heterogeneous_topk_kd():
    cfg = load_dense_config(REPO / "configs" / "pre2_dense_1_5b.yaml")

    assert "heterogeneous_tokenizer_topk_logit_kd" in cfg.training.objective.disabled_paths
    assert cfg.precision_policy["int4_training"] == "not_allowed"


def test_pre2_proxy_config_is_smaller_but_same_family():
    cfg = load_dense_config(REPO / "configs" / "pre2_dense_proxy_400m.yaml")

    assert cfg.model.hidden_dim == 1024
    assert cfg.model.attention.type == "gqa"
    assert 350_000_000 <= cfg.parameter_count_estimate() <= 410_000_000
    assert cfg.training.batch.global_batch_tokens == 1_048_576
    assert cfg.training_steps_for_tokens(cfg.training.token_budget.study_target) == 28_611


def test_pre2_compute_limited_ladder_configs_match_plan():
    canary = load_dense_config(REPO / "configs" / "pre2_dense_canary_110m.yaml")
    poc = load_dense_config(REPO / "configs" / "pre2_dense_poc_250m.yaml")
    proxy = load_dense_config(REPO / "configs" / "pre2_dense_proxy_400m.yaml")
    main = load_dense_config(REPO / "configs" / "pre2_dense_1_5b.yaml")

    assert canary.parameter_count_estimate() == 112_737_280
    assert poc.parameter_count_estimate() == 239_104_256
    assert (
        canary.parameter_count_estimate()
        < poc.parameter_count_estimate()
        < proxy.parameter_count_estimate()
        < main.parameter_count_estimate()
    )
    assert canary.model.attention.type == poc.model.attention.type == "gqa"
    assert canary.model.activation == poc.model.activation == "swiglu"
    assert canary.model.norm == poc.model.norm == "rmsnorm"
    assert canary.model.position.type == poc.model.position.type == "rope"
    assert canary.training.dtype == poc.training.dtype == "bf16"
    assert canary.training.token_budget.study_target == 1_000_000_000
    assert poc.training.token_budget.study_target == 10_000_000_000
    assert canary.training_steps_for_tokens(canary.training.token_budget.study_target) == 3815
    assert poc.training_steps_for_tokens(poc.training.token_budget.study_target) == 19074


@pytest.mark.parametrize(
    "path",
    [
        "configs/data/pre2_mix_stage1.yaml",
        "configs/data/pre2_mix_stage2.yaml",
        "configs/data/pre2_mix_anneal.yaml",
        "configs/data/pre2_mix_v0_5.yaml",
    ],
)
def test_pre2_data_mix_configs_parse(path: str):
    cfg = load_data_mix_config(REPO / path)

    assert cfg.status == "planning"
    assert cfg.target_tokens > 0
    assert cfg.synthetic_cap <= 0.10
    assert cfg.source_buckets


def test_pre2_target_share_mixes_sum_to_one():
    for path in [
        REPO / "configs" / "data" / "pre2_mix_stage2.yaml",
        REPO / "configs" / "data" / "pre2_mix_anneal.yaml",
        REPO / "configs" / "data" / "pre2_mix_v0_5.yaml",
    ]:
        cfg = load_data_mix_config(path)
        total = sum(bucket.target_share or 0.0 for bucket in cfg.source_buckets)
        assert total == pytest.approx(1.0)


def test_pre2_stage1_requires_manifest_metadata():
    cfg = load_data_mix_config(REPO / "configs" / "data" / "pre2_mix_stage1.yaml")

    assert cfg.required_metadata is not None
    for field in ["source", "source_version", "license", "tokenizer_hash", "document_hash"]:
        assert field in cfg.required_metadata


def test_pre2_v0_5_mix_is_two_source_10b_contract():
    cfg = load_data_mix_config(REPO / "configs" / "data" / "pre2_mix_v0_5.yaml")

    assert cfg.name == "pre2_mix_v0_5_poc_10b"
    assert cfg.target_tokens == 10_000_000_000
    assert cfg.synthetic_cap == 0.0
    assert [bucket.bucket for bucket in cfg.source_buckets] == [
        "high_quality_educational_web",
        "math_stem",
    ]
    assert [bucket.target_share for bucket in cfg.source_buckets] == [0.85, 0.15]


def test_pre1_flat_config_is_rejected_by_pre2_schema():
    with pytest.raises(Exception):
        load_dense_config(REPO / "configs" / "base_1b.yaml")
