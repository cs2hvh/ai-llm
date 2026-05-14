"""Single-process pre-2 checkpoint contract.

This is a smoke checkpoint format, not the final PyTorch Distributed
Checkpointing integration. It establishes the exact state we must carry into
the DCP implementation: model, optimizer, scheduler, data cursor, and RNG.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from myllm_pre2.config import DensePre2Config


CHECKPOINT_FORMAT_VERSION = 1
CHECKPOINT_FILENAME = "checkpoint.pt"
MANIFEST_FILENAME = "manifest.json"


@dataclass(frozen=True)
class Pre2ModelState:
    config_name: str
    parameter_count: int
    state_dict: dict[str, Any]


@dataclass(frozen=True)
class Pre2OptimizerState:
    optimizer_type: str
    state_dict: dict[str, Any]


@dataclass(frozen=True)
class Pre2SchedulerState:
    scheduler_type: str | None = None
    state_dict: dict[str, Any] | None = None
    last_lr: list[float] | None = None


@dataclass(frozen=True)
class Pre2DataCursor:
    next_sequence_id: int
    tokens_consumed: int
    epoch: int = 0

    def __post_init__(self) -> None:
        if self.next_sequence_id < 0:
            raise ValueError("next_sequence_id must be >= 0")
        if self.tokens_consumed < 0:
            raise ValueError("tokens_consumed must be >= 0")
        if self.epoch < 0:
            raise ValueError("epoch must be >= 0")


@dataclass(frozen=True)
class Pre2RngState:
    torch_cpu_state: Any
    torch_cuda_states: list[Any] = field(default_factory=list)
    python_state: Any | None = None
    numpy_state: Any | None = None


@dataclass(frozen=True)
class Pre2CheckpointPayload:
    format_version: int
    step: int
    model: Pre2ModelState
    optimizer: Pre2OptimizerState
    scheduler: Pre2SchedulerState
    data: Pre2DataCursor
    rng: Pre2RngState
    tokenizer_sha256: str | None = None
    config_digest: str | None = None

    def __post_init__(self) -> None:
        if self.format_version != CHECKPOINT_FORMAT_VERSION:
            raise ValueError(f"unsupported checkpoint format {self.format_version}")
        if self.step < 0:
            raise ValueError("step must be >= 0")


def capture_rng_state(*, include_cuda: bool = True) -> Pre2RngState:
    """Capture Python, NumPy, and Torch RNG state."""
    numpy_state = None
    try:
        import numpy as np

        numpy_state = np.random.get_state()
    except Exception:
        numpy_state = None

    cuda_states: list[Any] = []
    if include_cuda and torch.cuda.is_available():
        cuda_states = list(torch.cuda.get_rng_state_all())

    return Pre2RngState(
        torch_cpu_state=torch.get_rng_state(),
        torch_cuda_states=cuda_states,
        python_state=random.getstate(),
        numpy_state=numpy_state,
    )


def restore_rng_state(state: Pre2RngState) -> None:
    """Restore a state captured by ``capture_rng_state``."""
    torch.set_rng_state(state.torch_cpu_state)
    if state.torch_cuda_states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state.torch_cuda_states)
    if state.python_state is not None:
        random.setstate(state.python_state)
    if state.numpy_state is not None:
        try:
            import numpy as np

            np.random.set_state(state.numpy_state)
        except Exception:
            pass


def model_parameter_count(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def build_checkpoint_payload(
    *,
    cfg: DensePre2Config,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    data_cursor: Pre2DataCursor,
    scheduler: Any | None = None,
    tokenizer_sha256: str | None = None,
    config_digest: str | None = None,
) -> Pre2CheckpointPayload:
    scheduler_state = Pre2SchedulerState()
    if scheduler is not None:
        scheduler_state = Pre2SchedulerState(
            scheduler_type=type(scheduler).__name__,
            state_dict=scheduler.state_dict(),
            last_lr=list(scheduler.get_last_lr()) if hasattr(scheduler, "get_last_lr") else None,
        )

    return Pre2CheckpointPayload(
        format_version=CHECKPOINT_FORMAT_VERSION,
        step=step,
        model=Pre2ModelState(
            config_name=cfg.name,
            parameter_count=model_parameter_count(model),
            state_dict=model.state_dict(),
        ),
        optimizer=Pre2OptimizerState(
            optimizer_type=type(optimizer).__name__,
            state_dict=optimizer.state_dict(),
        ),
        scheduler=scheduler_state,
        data=data_cursor,
        rng=capture_rng_state(),
        tokenizer_sha256=tokenizer_sha256,
        config_digest=config_digest,
    )


def restore_checkpoint_payload(
    payload: Pre2CheckpointPayload,
    *,
    cfg: DensePre2Config,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    restore_rng: bool = True,
) -> Pre2DataCursor:
    """Restore model, optimizer, scheduler, and RNG from a smoke payload.

    Returns the data cursor carried by the checkpoint so callers can resume the
    matching dataloader position.
    """
    if payload.model.config_name != cfg.name:
        raise ValueError(
            "checkpoint config mismatch: "
            f"payload={payload.model.config_name!r}, cfg={cfg.name!r}"
        )

    expected_params = model_parameter_count(model)
    if payload.model.parameter_count != expected_params:
        raise ValueError(
            "checkpoint parameter-count mismatch: "
            f"payload={payload.model.parameter_count}, model={expected_params}"
        )

    model.load_state_dict(payload.model.state_dict)
    if optimizer is not None:
        optimizer.load_state_dict(payload.optimizer.state_dict)
    if scheduler is not None and payload.scheduler.state_dict is not None:
        scheduler.load_state_dict(payload.scheduler.state_dict)
    if restore_rng:
        restore_rng_state(payload.rng)
    return payload.data


def save_checkpoint(root: str | Path, payload: Pre2CheckpointPayload) -> None:
    """Atomically save a smoke checkpoint directory."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = root / CHECKPOINT_FILENAME
    tmp_path = root / f"{CHECKPOINT_FILENAME}.tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, checkpoint_path)

    manifest = {
        "format_version": payload.format_version,
        "step": payload.step,
        "config_name": payload.model.config_name,
        "parameter_count": payload.model.parameter_count,
        "next_sequence_id": payload.data.next_sequence_id,
        "tokens_consumed": payload.data.tokens_consumed,
        "epoch": payload.data.epoch,
        "tokenizer_sha256": payload.tokenizer_sha256,
        "config_digest": payload.config_digest,
        "checkpoint_file": CHECKPOINT_FILENAME,
    }
    manifest_path = root / MANIFEST_FILENAME
    tmp_manifest = root / f"{MANIFEST_FILENAME}.tmp"
    tmp_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp_manifest, manifest_path)


def load_checkpoint(root: str | Path, *, map_location: str | torch.device = "cpu") -> Pre2CheckpointPayload:
    """Load a smoke checkpoint payload."""
    path = Path(root) / CHECKPOINT_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"checkpoint missing: {path}")
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, Pre2CheckpointPayload):
        raise TypeError(f"unexpected checkpoint payload type: {type(payload)!r}")
    return payload
