from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.pre2_v0_5_build_commands import build_commands


REPO = Path(__file__).resolve().parents[1]


def test_pre2_v0_5_build_mix_sources_match_contract():
    with open(REPO / "configs" / "data" / "pre2_v0_5_build_mix.yaml", encoding="utf-8") as f:
        build_mix = yaml.safe_load(f)

    sources = build_mix["sources"]
    assert sum(float(source["share"]) for source in sources) == pytest.approx(1.0)
    assert [source["source_id"] for source in sources] == [
        "fineweb_edu_v0_5",
        "open_web_math_v0_5",
    ]
    assert sum(int(source["target_tokens"]) for source in sources) == 10_000_000_000
    assert sources[0]["config_name"] == "sample-10BT"


def test_pre2_v0_5_build_commands_pin_revision_and_token_targets():
    commands = build_commands(
        repo=REPO,
        build_mix_path=REPO / "configs" / "data" / "pre2_v0_5_build_mix.yaml",
        output_root="/data/pre2/v0_5/sources",
        tokenizer_path="artifacts/tokenizer_v1.json",
        sequence_length=8193,
        sequences_per_shard=65536,
        production=True,
        shell="posix",
    )

    assert len(commands) == 2
    assert "--hf-revision 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9" in commands[0]
    assert "--target-tokens 8500000000" in commands[0]
    assert "--hf-revision fde8ef8de2300f5e778f56261843dab89f230815" in commands[1]
    assert "--target-tokens 1500000000" in commands[1]
    assert all("--production" in command for command in commands)
