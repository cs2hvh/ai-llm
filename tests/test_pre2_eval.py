from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from myllm_pre2.config import load_dense_config  # noqa: E402
from myllm_pre2.eval import (  # noqa: E402
    NextTokenEvalExample,
    build_next_token_predict_fn,
    run_checkpoint_next_token_eval,
)


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "pre2_train.py"
SPEC = importlib.util.spec_from_file_location("pre2_train", SCRIPT)
pre2_train = importlib.util.module_from_spec(SPEC)
sys.modules["pre2_train"] = pre2_train
SPEC.loader.exec_module(pre2_train)


def _make_checkpoint(tmp_path: Path):
    base_cfg = load_dense_config(REPO / "configs" / "pre2_dense_canary_110m.yaml")
    tiny_cfg = pre2_train.make_tiny_smoke_config(base_cfg)
    checkpoint_dir = tmp_path / "ckpt"
    pre2_train.run_synthetic_smoke(
        tiny_cfg,
        steps=1,
        batch_size=1,
        sequence_length=8,
        device="cpu",
        checkpoint_dir=checkpoint_dir,
    )
    return tiny_cfg, checkpoint_dir


def test_pre2_checkpoint_next_token_eval_runs_real_prediction_path(tmp_path: Path):
    cfg, checkpoint_dir = _make_checkpoint(tmp_path)

    result = run_checkpoint_next_token_eval(
        cfg,
        checkpoint_dir,
        [
            NextTokenEvalExample(prompt_ids=[1, 2, 3], target_id=4),
            NextTokenEvalExample(prompt_ids=[5, 6], target_id=7),
        ],
        device="cpu",
    )

    assert result.checkpoint_step == 1
    assert result.num_examples == 2
    assert 0.0 <= result.accuracy <= 1.0
    assert result.average_nll > 0
    assert len(result.predictions) == 2
    for prediction in result.predictions:
        assert 0 <= prediction.predicted_id < cfg.model.tokenizer.planning_vocab_size()


def test_pre2_next_token_predict_fn_is_reusable(tmp_path: Path):
    cfg, checkpoint_dir = _make_checkpoint(tmp_path)

    predict = build_next_token_predict_fn(cfg, checkpoint_dir, device="cpu")
    first = predict([1, 2, 3])
    second = predict([1, 2, 3])

    assert first == second
    assert 0 <= first < cfg.model.tokenizer.planning_vocab_size()


def test_pre2_eval_rejects_empty_prompt(tmp_path: Path):
    cfg, checkpoint_dir = _make_checkpoint(tmp_path)

    with pytest.raises(ValueError, match="prompt_ids"):
        run_checkpoint_next_token_eval(
            cfg,
            checkpoint_dir,
            [NextTokenEvalExample(prompt_ids=[], target_id=1)],
            device="cpu",
        )
