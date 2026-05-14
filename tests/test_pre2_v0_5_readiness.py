from __future__ import annotations

from pathlib import Path

from scripts.pre2_v0_5_readiness import build_payload


REPO = Path(__file__).resolve().parents[1]


def test_pre2_v0_5_readiness_blocks_missing_tokenizer():
    payload, exit_code = build_payload(
        mix_path=REPO / "configs" / "data" / "pre2_mix_v0_5.yaml",
        registry_path=REPO / "configs" / "data" / "pre2_source_registry.yaml",
        build_mix_path=REPO / "configs" / "data" / "pre2_v0_5_build_mix.yaml",
        tokenizer_path=REPO / "artifacts" / "definitely-missing-tokenizer.json",
        stage="poc",
        allow_missing_tokenizer=False,
        allow_missing_decontam=False,
    )

    assert exit_code == 2
    assert payload["ready"] is False
    assert any("tokenizer artifact missing" in issue for issue in payload["blocking_issues"])
    assert any("decontamination index missing" in issue for issue in payload["blocking_issues"])
    assert payload["storage"]["recommended_local_nvme_gb"] == 500


def test_pre2_v0_5_readiness_passes_with_tokenizer_artifact(tmp_path):
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text('{"model":"smoke"}', encoding="utf-8")

    payload, exit_code = build_payload(
        mix_path=REPO / "configs" / "data" / "pre2_mix_v0_5.yaml",
        registry_path=REPO / "configs" / "data" / "pre2_source_registry.yaml",
        build_mix_path=REPO / "configs" / "data" / "pre2_v0_5_build_mix.yaml",
        tokenizer_path=tokenizer,
        stage="poc",
        allow_missing_tokenizer=False,
        allow_missing_decontam=True,
    )

    assert exit_code == 0
    assert payload["ready"] is True
    assert payload["target_tokens"] == 10_000_000_000
    assert payload["tokenizer"]["exists"] is True
    assert len(payload["tokenizer"]["sha256"]) == 64
