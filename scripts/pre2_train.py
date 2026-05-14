#!/usr/bin/env python3
"""Pre-2 synthetic training smoke entrypoint.

This is not the main TorchTitan trainer. It is a deliberately small CPU/1-GPU
smoke that proves the pre-2 config, model, loss, optimizer, and backward pass
work together before distributed training is added.
"""
from __future__ import annotations

import argparse
import sys
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from myllm_pre2.checkpoint import (
    Pre2DataCursor,
    build_checkpoint_payload,
    load_checkpoint,
    restore_checkpoint_payload,
    save_checkpoint,
)
from myllm_pre2.config import DensePre2Config, load_dense_config
from myllm_pre2.data import PackedTorchDataLoader
from myllm_pre2.model import build_dense_lm


def _select_device(device: str):
    import torch

    selected_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
    if selected_device == "auto":
        selected_device = "cpu"
    return torch.device(selected_device)


def _resolve_precision(cfg: DensePre2Config, precision: str):
    import torch

    requested = cfg.training.dtype if precision == "config" else precision
    if requested == "fp32":
        return "fp32", None
    if requested == "bf16":
        return "bf16", torch.bfloat16
    raise ValueError(f"unsupported pre-2 smoke precision: {precision}")


def _autocast_context(torch_device, autocast_dtype):
    import torch

    if autocast_dtype is None:
        return nullcontext()
    return torch.autocast(device_type=torch_device.type, dtype=autocast_dtype)


def _build_optimizer(cfg: DensePre2Config, model):
    import torch

    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.scheduler.peak_lr or 2.0e-4,
        betas=(cfg.training.optimizer.beta1, cfg.training.optimizer.beta2),
        weight_decay=cfg.training.optimizer.weight_decay,
        eps=cfg.training.optimizer.eps or 1.0e-8,
    )


def _backward_step(cfg: DensePre2Config, model, optimizer, loss) -> float:
    import torch

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_clip = cfg.training.optimizer.grad_clip_global_norm
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return float(grad_norm.detach().cpu())


def _cuda_peak_memory_mb(torch_device) -> float | None:
    import torch

    if torch_device.type != "cuda":
        return None
    return float(torch.cuda.max_memory_allocated(torch_device) / (1024 * 1024))


def make_tiny_smoke_config(base_cfg: DensePre2Config) -> DensePre2Config:
    """Derive a tiny executable config from the real pre-2 contract."""
    data = base_cfg.model_dump(mode="python")
    data = deepcopy(data)
    data["name"] = "myllm-pre2-dense-tiny-smoke"
    data["model"].update(
        {
            "parameter_target": 100_000,
            "layers": 2,
            "hidden_dim": 64,
            "ffn_dim": 192,
            "attention": {
                "type": "gqa",
                "num_heads": 4,
                "num_kv_heads": 2,
                "head_dim": 16,
            },
            "tokenizer": {
                "candidates": [256],
                "current_reference_vocab_size": 256,
            },
            "context": {
                "foundation_length": 32,
                "continuation_lengths": [],
            },
        }
    )
    data["training"]["batch"] = {
        "global_batch_tokens": 64,
        "sequence_length": 32,
    }
    data["training"]["token_budget"] = {"smoke": 2048}
    data["parallelism"] = {"strategy": "single_process_smoke"}
    return DensePre2Config.model_validate(data)


def run_synthetic_smoke(
    cfg: DensePre2Config,
    *,
    steps: int = 2,
    batch_size: int = 2,
    sequence_length: int | None = None,
    device: str = "auto",
    precision: str = "fp32",
    checkpoint_dir: str | Path | None = None,
    resume_from_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    import torch

    if steps <= 0:
        raise ValueError("steps must be > 0")

    torch_device = _select_device(device)
    precision_name, autocast_dtype = _resolve_precision(cfg, precision)
    if torch_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(torch_device)

    model = build_dense_lm(cfg).to(torch_device)
    model.train()
    optimizer = _build_optimizer(cfg, model)

    start_step = 0
    tokens_consumed = 0
    next_sequence_id = 0
    if resume_from_checkpoint is not None:
        payload = load_checkpoint(resume_from_checkpoint, map_location=torch_device)
        data_cursor = restore_checkpoint_payload(
            payload,
            cfg=cfg,
            model=model,
            optimizer=optimizer,
            restore_rng=True,
        )
        start_step = payload.step
        tokens_consumed = data_cursor.tokens_consumed
        next_sequence_id = data_cursor.next_sequence_id

    vocab = cfg.model.tokenizer.planning_vocab_size()
    seq_len = sequence_length or cfg.training.batch.sequence_length or cfg.model.context.foundation_length
    last_loss = None
    last_grad_norm = None
    for _ in range(steps):
        tokens = torch.randint(0, vocab, (batch_size, seq_len + 1), device=torch_device)
        input_ids = tokens[:, :-1].contiguous()
        labels = tokens[:, 1:].contiguous()
        loss_mask = torch.ones_like(labels, dtype=torch.float32, device=torch_device)
        with _autocast_context(torch_device, autocast_dtype):
            out = model(input_ids, labels=labels, loss_mask=loss_mask)
        if out.loss is None or not torch.isfinite(out.loss):
            raise RuntimeError("pre-2 synthetic smoke produced a non-finite loss")
        last_grad_norm = _backward_step(cfg, model, optimizer, out.loss)
        last_loss = float(out.loss.detach().cpu())
        tokens_consumed += batch_size * seq_len
        next_sequence_id += batch_size

    if checkpoint_dir is not None:
        payload = build_checkpoint_payload(
            cfg=cfg,
            model=model,
            optimizer=optimizer,
            step=start_step + steps,
            data_cursor=Pre2DataCursor(
                next_sequence_id=next_sequence_id,
                tokens_consumed=tokens_consumed,
            ),
            tokenizer_sha256="synthetic-smoke",
        )
        save_checkpoint(checkpoint_dir, payload)

    result = {
        "name": cfg.name,
        "device": str(torch_device),
        "steps": steps,
        "start_step": start_step,
        "total_steps": start_step + steps,
        "batch_size": batch_size,
        "sequence_length": seq_len,
        "precision": precision_name,
        "grad_clip_global_norm": cfg.training.optimizer.grad_clip_global_norm,
        "last_grad_norm": last_grad_norm,
        "tokens_consumed": tokens_consumed,
        "last_lr": optimizer.param_groups[0]["lr"],
        "peak_cuda_memory_mb": _cuda_peak_memory_mb(torch_device),
        "parameter_count": model.parameter_count(),
        "last_loss": last_loss,
    }
    if checkpoint_dir is not None:
        result["checkpoint_dir"] = str(checkpoint_dir)
    if resume_from_checkpoint is not None:
        result["resumed_from_checkpoint"] = str(resume_from_checkpoint)
    return result


def run_packed_corpus_smoke(
    cfg: DensePre2Config,
    *,
    packed_corpus_root: str | Path,
    steps: int = 2,
    batch_size: int = 2,
    device: str = "auto",
    precision: str = "fp32",
    checkpoint_dir: str | Path | None = None,
    resume_from_checkpoint: str | Path | None = None,
    expected_tokenizer_sha256: str | None = None,
    start_sequence_id: int | None = None,
) -> dict[str, Any]:
    """Run a tiny train smoke against the packed-corpus PyTorch adapter."""
    import torch

    if steps <= 0:
        raise ValueError("steps must be > 0")

    torch_device = _select_device(device)
    precision_name, autocast_dtype = _resolve_precision(cfg, precision)
    if torch_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(torch_device)

    model = build_dense_lm(cfg).to(torch_device)
    model.train()
    optimizer = _build_optimizer(cfg, model)

    start_step = 0
    tokens_consumed = 0
    resume_sequence_id = 0
    if resume_from_checkpoint is not None:
        payload = load_checkpoint(resume_from_checkpoint, map_location=torch_device)
        data_cursor = restore_checkpoint_payload(
            payload,
            cfg=cfg,
            model=model,
            optimizer=optimizer,
            restore_rng=True,
        )
        start_step = payload.step
        tokens_consumed = data_cursor.tokens_consumed
        resume_sequence_id = data_cursor.next_sequence_id
    if start_sequence_id is None:
        start_sequence_id = resume_sequence_id

    loader = PackedTorchDataLoader(
        root=packed_corpus_root,
        batch_size=batch_size,
        device=torch_device,
        expected_tokenizer_sha256=expected_tokenizer_sha256,
    )
    batches = loader.iter_batches(start_sequence_id=start_sequence_id)

    last_loss = None
    next_sequence_id = start_sequence_id
    completed_steps = 0
    last_grad_norm = None
    for _ in range(steps):
        try:
            batch = next(batches)
        except StopIteration:
            break

        with _autocast_context(torch_device, autocast_dtype):
            out = model(batch.input_ids, labels=batch.labels, loss_mask=batch.loss_mask)
        if out.loss is None or not torch.isfinite(out.loss):
            raise RuntimeError("pre-2 packed-corpus smoke produced a non-finite loss")
        last_grad_norm = _backward_step(cfg, model, optimizer, out.loss)

        last_loss = float(out.loss.detach().cpu())
        tokens_consumed += int(batch.input_ids.numel())
        next_sequence_id = batch.next_sequence_id
        completed_steps += 1

    if completed_steps == 0:
        raise RuntimeError("packed corpus yielded no trainable batches")

    tokenizer_sha256 = getattr(getattr(loader.reader, "manifest", None), "tokenizer_sha256", None)
    if checkpoint_dir is not None:
        payload = build_checkpoint_payload(
            cfg=cfg,
            model=model,
            optimizer=optimizer,
            step=start_step + completed_steps,
            data_cursor=Pre2DataCursor(
                next_sequence_id=next_sequence_id,
                tokens_consumed=tokens_consumed,
            ),
            tokenizer_sha256=tokenizer_sha256,
        )
        save_checkpoint(checkpoint_dir, payload)

    result = {
        "name": cfg.name,
        "mode": "packed_corpus",
        "device": str(torch_device),
        "steps": completed_steps,
        "start_step": start_step,
        "total_steps": start_step + completed_steps,
        "batch_size": batch_size,
        "next_sequence_id": next_sequence_id,
        "tokens_consumed": tokens_consumed,
        "precision": precision_name,
        "grad_clip_global_norm": cfg.training.optimizer.grad_clip_global_norm,
        "last_grad_norm": last_grad_norm,
        "last_lr": optimizer.param_groups[0]["lr"],
        "peak_cuda_memory_mb": _cuda_peak_memory_mb(torch_device),
        "parameter_count": model.parameter_count(),
        "last_loss": last_loss,
        "tokenizer_sha256": tokenizer_sha256,
    }
    if checkpoint_dir is not None:
        result["checkpoint_dir"] = str(checkpoint_dir)
    if resume_from_checkpoint is not None:
        result["resumed_from_checkpoint"] = str(resume_from_checkpoint)
    return result


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-config",
        default=str(repo / "configs" / "pre2_dense_1_5b.yaml"),
        help="Pre-2 model config to validate before deriving the tiny smoke config.",
    )
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--precision",
        default="fp32",
        choices=["fp32", "bf16", "config"],
        help="Smoke precision. Use 'config' to follow training.dtype from the pre-2 config.",
    )
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Optional smoke checkpoint directory to restore before running additional steps.",
    )
    parser.add_argument(
        "--packed-corpus-root",
        default=None,
        help="Optional packed corpus root. If supplied, train on packed batches instead of synthetic tokens.",
    )
    parser.add_argument("--expected-tokenizer-sha256", default=None)
    parser.add_argument("--start-sequence-id", type=int, default=None)
    parser.add_argument(
        "--full-config",
        action="store_true",
        help="Use the supplied config directly. Dangerous on CPU for the 1.5B config.",
    )
    args = parser.parse_args()

    base_cfg = load_dense_config(args.model_config)
    cfg = base_cfg if args.full_config else make_tiny_smoke_config(base_cfg)
    if args.packed_corpus_root is not None:
        result = run_packed_corpus_smoke(
            cfg,
            packed_corpus_root=args.packed_corpus_root,
            steps=args.steps,
            batch_size=args.batch_size,
            device=args.device,
            precision=args.precision,
            checkpoint_dir=args.checkpoint_dir,
            resume_from_checkpoint=args.resume_from_checkpoint,
            expected_tokenizer_sha256=args.expected_tokenizer_sha256,
            start_sequence_id=args.start_sequence_id,
        )
    else:
        result = run_synthetic_smoke(
            cfg,
            steps=args.steps,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            device=args.device,
            precision=args.precision,
            checkpoint_dir=args.checkpoint_dir,
            resume_from_checkpoint=args.resume_from_checkpoint,
        )
    print(yaml.safe_dump(result, sort_keys=False).strip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
