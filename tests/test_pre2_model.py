from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional

from myllm_pre2.config import DensePre2Config  # noqa: E402
from myllm_pre2.model import build_dense_lm  # noqa: E402


def _tiny_cfg(z_loss_coef: float | None = 1.0e-4) -> DensePre2Config:
    return DensePre2Config.model_validate(
        {
            "schema_version": "0.1",
            "status": "planning",
            "name": "myllm-pre2-dense-tiny-test",
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
                "objective": {"next_token_ce": True, "z_loss_coef": z_loss_coef},
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
            "parallelism": {"strategy": "single_gpu_or_fsdp2"},
        }
    )


def test_pre2_tiny_model_forward_loss_and_backward():
    torch.manual_seed(1234)
    cfg = _tiny_cfg()
    model = build_dense_lm(cfg)
    tokens = torch.randint(0, 128, (2, 17))
    input_ids = tokens[:, :-1]
    labels = tokens[:, 1:]

    out = model(input_ids, labels=labels, loss_mask=torch.ones_like(labels, dtype=torch.float32))

    assert out.logits.shape == (2, 16, 128)
    assert out.loss is not None
    assert torch.isfinite(out.loss)
    out.loss.backward()
    assert model.token_embedding.weight.grad is not None


def test_pre2_model_ties_embeddings():
    model = build_dense_lm(_tiny_cfg())

    assert model.lm_head.weight is model.token_embedding.weight


def test_pre2_model_uses_aligned_labels_without_internal_shift():
    torch.manual_seed(1234)
    model = build_dense_lm(_tiny_cfg(z_loss_coef=0.0))
    input_ids = torch.randint(0, 128, (2, 16))
    labels = torch.randint(0, 128, (2, 16))

    out = model(input_ids, labels=labels)
    expected = F.cross_entropy(out.logits.view(-1, 128), labels.view(-1))

    assert out.loss is not None
    assert torch.allclose(out.loss, expected)


def test_pre2_model_respects_loss_mask_for_ce_and_z_loss():
    torch.manual_seed(1234)
    z_loss_coef = 1.0e-4
    model = build_dense_lm(_tiny_cfg(z_loss_coef=z_loss_coef))
    input_ids = torch.randint(0, 128, (2, 16))
    labels = torch.randint(0, 128, (2, 16))
    loss_mask = torch.ones((2, 16), dtype=torch.float32)
    loss_mask[0, 1] = 0.0
    loss_mask[1, 3] = 0.0

    out = model(input_ids, labels=labels, loss_mask=loss_mask)
    ce = F.cross_entropy(out.logits.view(-1, 128), labels.view(-1), reduction="none").view_as(labels)
    z_loss = torch.logsumexp(out.logits, dim=-1).pow(2)
    expected = ((ce + z_loss_coef * z_loss) * loss_mask).sum() / loss_mask.sum()

    assert out.loss is not None
    assert torch.allclose(out.loss, expected)


def test_pre2_model_rejects_loss_mask_shape_mismatch():
    model = build_dense_lm(_tiny_cfg())
    input_ids = torch.randint(0, 128, (1, 8))
    labels = torch.randint(0, 128, (1, 8))

    with pytest.raises(ValueError, match="loss_mask"):
        model(input_ids, labels=labels, loss_mask=torch.ones(1, 7))


def test_pre2_model_rejects_topk_kd_payload():
    model = build_dense_lm(_tiny_cfg())
    input_ids = torch.randint(0, 128, (1, 8))

    with pytest.raises(ValueError, match="heterogeneous-tokenizer top-K"):
        model(
            input_ids,
            teacher_topk_logits=torch.zeros(1, 1, 8, 4),
            teacher_topk_indices=torch.zeros(1, 1, 8, 4, dtype=torch.long),
        )
