from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from myllm_pre2.checkpoint import (  # noqa: E402
    Pre2DataCursor,
    build_checkpoint_payload,
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
    restore_checkpoint_payload,
    save_checkpoint,
)
from myllm_pre2.config import DensePre2Config  # noqa: E402
from myllm_pre2.model import build_dense_lm  # noqa: E402


def _tiny_cfg() -> DensePre2Config:
    return DensePre2Config.model_validate(
        {
            "schema_version": "0.1",
            "status": "planning",
            "name": "myllm-pre2-dense-checkpoint-test",
            "stack": "torchtitan",
            "model": {
                "architecture": "decoder_only_transformer",
                "parameter_target": 100_000,
                "layers": 2,
                "hidden_dim": 32,
                "ffn_dim": 96,
                "activation": "swiglu",
                "norm": "rmsnorm",
                "norm_eps": 1.0e-5,
                "qk_norm": True,
                "attention": {
                    "type": "gqa",
                    "num_heads": 4,
                    "num_kv_heads": 2,
                    "head_dim": 8,
                },
                "position": {"type": "rope", "rope_base": 10000},
                "embeddings": {"tied": True},
                "tokenizer": {"candidates": [128], "current_reference_vocab_size": 128},
                "context": {"foundation_length": 16},
            },
            "training": {
                "dtype": "bf16",
                "objective": {"next_token_ce": True, "z_loss_coef": 1.0e-4},
                "optimizer": {
                    "type": "adamw",
                    "beta1": 0.9,
                    "beta2": 0.95,
                    "weight_decay": 0.1,
                    "eps": 1.0e-8,
                    "grad_clip_global_norm": 1.0,
                },
                "scheduler": {
                    "type": "wsd",
                    "peak_lr": 2.0e-4,
                    "warmup_steps_range": [10, 20],
                    "decay_fraction": 0.1,
                },
                "batch": {"global_batch_tokens": 128, "sequence_length": 16},
                "token_budget": {"smoke": 1000},
            },
            "parallelism": {"strategy": "single_process_smoke"},
        }
    )


def _one_step(model, optimizer) -> float:
    tokens = torch.randint(0, 128, (2, 17))
    input_ids = tokens[:, :-1]
    labels = tokens[:, 1:]
    loss_mask = torch.ones_like(labels, dtype=torch.float32)
    out = model(input_ids, labels=labels, loss_mask=loss_mask)
    optimizer.zero_grad(set_to_none=True)
    out.loss.backward()
    optimizer.step()
    return float(out.loss.detach())


def _clone_model_state(model) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


def _assert_model_states_equal(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> None:
    assert left.keys() == right.keys()
    for name in left:
        assert torch.equal(left[name], right[name]), name


def _assert_optimizer_states_equal(left: dict, right: dict) -> None:
    assert left.keys() == right.keys()
    assert left["param_groups"] == right["param_groups"]
    assert left["state"].keys() == right["state"].keys()
    for param_id, left_state in left["state"].items():
        right_state = right["state"][param_id]
        assert left_state.keys() == right_state.keys()
        for key, left_value in left_state.items():
            right_value = right_state[key]
            if torch.is_tensor(left_value):
                assert torch.equal(left_value, right_value), (param_id, key)
            else:
                assert left_value == right_value


def test_pre2_rng_capture_restore_is_exact():
    torch.manual_seed(1234)
    state = capture_rng_state(include_cuda=False)
    first = torch.randint(0, 1000, (8,))
    restore_rng_state(state)
    second = torch.randint(0, 1000, (8,))

    assert torch.equal(first, second)


def test_pre2_smoke_checkpoint_round_trip(tmp_path: Path):
    torch.manual_seed(1234)
    cfg = _tiny_cfg()
    model = build_dense_lm(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-4)
    _one_step(model, optimizer)
    saved_model_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    saved_optimizer_state = deepcopy(optimizer.state_dict())

    payload = build_checkpoint_payload(
        cfg=cfg,
        model=model,
        optimizer=optimizer,
        step=7,
        data_cursor=Pre2DataCursor(next_sequence_id=14, tokens_consumed=224),
        tokenizer_sha256="tok-sha",
    )
    save_checkpoint(tmp_path / "ckpt", payload)

    loaded = load_checkpoint(tmp_path / "ckpt")
    assert loaded.step == 7
    assert loaded.data.next_sequence_id == 14
    assert loaded.data.tokens_consumed == 224
    assert loaded.tokenizer_sha256 == "tok-sha"

    restored_model = build_dense_lm(cfg)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=2.0e-4)
    restored_model.load_state_dict(loaded.model.state_dict)
    restored_optimizer.load_state_dict(loaded.optimizer.state_dict)

    for name, tensor in restored_model.state_dict().items():
        assert torch.equal(tensor, saved_model_state[name])
    assert restored_optimizer.state_dict()["param_groups"] == saved_optimizer_state["param_groups"]
    assert (tmp_path / "ckpt" / "manifest.json").exists()


def test_pre2_smoke_checkpoint_resume_matches_uninterrupted_run(tmp_path: Path):
    cfg = _tiny_cfg()

    torch.manual_seed(2026)
    reference_model = build_dense_lm(cfg)
    reference_optimizer = torch.optim.AdamW(reference_model.parameters(), lr=2.0e-4)
    _one_step(reference_model, reference_optimizer)
    reference_loss = _one_step(reference_model, reference_optimizer)
    reference_model_state = _clone_model_state(reference_model)
    reference_optimizer_state = deepcopy(reference_optimizer.state_dict())

    torch.manual_seed(2026)
    interrupted_model = build_dense_lm(cfg)
    interrupted_optimizer = torch.optim.AdamW(interrupted_model.parameters(), lr=2.0e-4)
    _one_step(interrupted_model, interrupted_optimizer)
    payload = build_checkpoint_payload(
        cfg=cfg,
        model=interrupted_model,
        optimizer=interrupted_optimizer,
        step=1,
        data_cursor=Pre2DataCursor(next_sequence_id=2, tokens_consumed=32),
        tokenizer_sha256="tok-sha",
    )
    save_checkpoint(tmp_path / "ckpt", payload)

    torch.manual_seed(9999)
    restored_model = build_dense_lm(cfg)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=2.0e-4)
    loaded = load_checkpoint(tmp_path / "ckpt")
    cursor = restore_checkpoint_payload(
        loaded,
        cfg=cfg,
        model=restored_model,
        optimizer=restored_optimizer,
        restore_rng=True,
    )
    resumed_loss = _one_step(restored_model, restored_optimizer)

    assert loaded.step == 1
    assert cursor.next_sequence_id == 2
    assert cursor.tokens_consumed == 32
    assert resumed_loss == reference_loss
    _assert_model_states_equal(_clone_model_state(restored_model), reference_model_state)
    _assert_optimizer_states_equal(restored_optimizer.state_dict(), reference_optimizer_state)


def test_pre2_smoke_checkpoint_restore_rejects_config_mismatch():
    cfg = _tiny_cfg()
    model = build_dense_lm(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-4)
    payload = build_checkpoint_payload(
        cfg=cfg,
        model=model,
        optimizer=optimizer,
        step=0,
        data_cursor=Pre2DataCursor(next_sequence_id=0, tokens_consumed=0),
    )
    other_cfg = cfg.model_copy(update={"name": "other-pre2-config"})

    with pytest.raises(ValueError, match="config mismatch"):
        restore_checkpoint_payload(
            payload,
            cfg=other_cfg,
            model=model,
            optimizer=optimizer,
        )
