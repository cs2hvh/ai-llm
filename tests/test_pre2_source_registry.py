from __future__ import annotations

from pathlib import Path

import pytest

from myllm_pre2.data.source_registry import SourceRegistry, load_source_registry


REPO = Path(__file__).resolve().parents[1]


def test_pre2_source_registry_parses_and_counts_statuses():
    registry = load_source_registry(REPO / "configs" / "data" / "pre2_source_registry.yaml")

    assert registry.name == "pre2_source_registry_v0"
    counts = registry.status_counts()
    assert counts["approved"] == 3
    assert counts["needs_review"] >= 8
    assert counts["blocked"] == 1
    assert counts["excluded"] == 1
    assert "high_quality_educational_web" in registry.bucket_counts()


def test_pre2_source_registry_stage_gate_is_fail_closed():
    registry = load_source_registry(REPO / "configs" / "data" / "pre2_source_registry.yaml")

    assert [source.source_id for source in registry.sources_for_stage("canary")] == [
        "pre2_tiny_internal_fixture"
    ]
    assert [source.source_id for source in registry.sources_for_stage("poc")] == [
        "fineweb_edu_v0_5",
        "open_web_math_v0_5",
    ]
    assert registry.estimated_tokens_for_stage("canary") == 100_000
    assert registry.estimated_tokens_for_stage("poc") >= 10_000_000_000


def test_pre2_source_registry_rejects_duplicate_ids():
    data = {
        "schema_version": "0.1",
        "status": "planning",
        "name": "bad",
        "sources": [
            {
                "source_id": "dup",
                "display_name": "A",
                "source_type": "fixture",
                "buckets": ["fixture"],
                "license_status": "approved",
                "license_expression": "Apache-2.0",
                "revision": "r1",
            },
            {
                "source_id": "dup",
                "display_name": "B",
                "source_type": "fixture",
                "buckets": ["fixture"],
                "license_status": "needs_review",
            },
        ],
    }

    with pytest.raises(ValueError, match="duplicate"):
        SourceRegistry.model_validate(data)


def test_pre2_source_registry_rejects_unapproved_allowed_stage():
    data = {
        "schema_version": "0.1",
        "status": "planning",
        "name": "bad",
        "sources": [
            {
                "source_id": "candidate",
                "display_name": "Candidate",
                "source_type": "web",
                "buckets": ["web"],
                "license_status": "needs_review",
                "allowed_stages": ["poc"],
            }
        ],
    }

    with pytest.raises(ValueError, match="only approved"):
        SourceRegistry.model_validate(data)
