"""Shared checkpoint-load + greedy-decode for downstream eval.

Built during Round B4 (2026-05-16) so the release scorecard's
``predict_fn`` doesn't need to duplicate the (already-debugged)
checkpoint-restore + forward path from ``scripts/generate.py``.

Surface
-------
- ``load_checkpoint_for_inference(...)``: returns a ``LoadedCheckpoint``
  bundle (model + trainable_vars + non_trainable_vars + model_cfg +
  tokenizer + special-token ids + JIT'd forward). Cross-mesh restore
  is handled (G6 fix); single-device inference pods can load
  checkpoints saved on multi-device meshes.

- ``build_greedy_predict_fn(...)``: convenience that loads then returns
  a ``Callable[[str], str]`` — input prompt, output continuation.
  Greedy (argmax) decoding, deterministic. Used by the release
  scorecard. For temperature/top-p decoding see ``scripts/generate.py``.

These helpers are JAX-heavy. Importing the module is cheap, but the
load call materialises the model on device.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from myllm.utils import get_logger

log = get_logger(__name__)


@dataclass
class LoadedCheckpoint:
    """Everything an inference call needs.

    ``forward_jit`` takes ``(trainable, non_trainable, input_ids)`` and
    returns logits of shape ``[B, S, V]``. ``ctx_length`` is the model's
    context window. ``pad_id`` / ``eos_id`` are looked up from the
    tokenizer's special-token map.
    """
    model: Any
    trainable: Any
    non_trainable: Any
    model_cfg: Any
    tokenizer: Any
    forward_jit: Callable[..., Any]
    ctx_length: int
    pad_id: int
    eos_id: int
    step: int


def _step_number_from_path(path: Path) -> int:
    name = path.name
    if not name.startswith("step-"):
        raise ValueError(f"Expected path ending in 'step-XXXXX', got {path}")
    return int(name[len("step-"):])


def load_checkpoint_for_inference(
    *,
    model_config_path: Path,
    tokenizer_path: Path,
    checkpoint_root: Path,
    checkpoint_step: int | None = None,
    tokenizer_key: str | None = None,
) -> LoadedCheckpoint:
    """Build the model from config + restore weights from a checkpoint.

    Args:
        model_config_path: Path to a ModelConfig yaml.
        tokenizer_path: Local path to the tokenizer.json. If missing,
            and ``tokenizer_key`` is set, it's fetched from R2.
        checkpoint_root: Path containing ``step-XXXXX/`` subdirs.
        checkpoint_step: Specific step to load. Default: latest complete.
        tokenizer_key: Optional R2 key for the tokenizer.

    Returns:
        ``LoadedCheckpoint`` bundle ready for inference calls.
    """
    import jax
    from myllm.data.special_tokens import (
        SpecialTokens,
        verify_tokenizer_has_required,
    )
    from myllm.data.tokenize import load_tokenizer
    from myllm.model.config import ModelConfig
    from myllm.training.checkpoint import (
        CheckpointConfig,
        CheckpointManager,
    )
    from myllm.training.optimizer import OptimizerConfig
    from myllm.training.state_init import (
        init_model_and_optimizer,
        initial_train_state,
    )
    from myllm.utils.storage import ensure_tokenizer_local

    checkpoint_root = Path(checkpoint_root).resolve()
    if not checkpoint_root.exists():
        raise FileNotFoundError(f"checkpoint_root does not exist: {checkpoint_root}")

    # Resolve step: explicit > latest. The CheckpointManager looks for
    # ``step-NNNNNNNNN/manifest.json``; we accept either a parent dir or
    # a direct step dir.
    if (checkpoint_root / "manifest.json").exists():
        # Caller passed the step-dir directly.
        step = _step_number_from_path(checkpoint_root)
        ckpt_parent = checkpoint_root.parent
    else:
        ckpt_parent = checkpoint_root
        mgr_probe = CheckpointManager(
            CheckpointConfig(
                root=str(ckpt_parent), keep_last_n=1, keep_every_n=10**9,
                r2_prefix=None,
            )
        )
        if checkpoint_step is not None:
            step = int(checkpoint_step)
        else:
            latest = mgr_probe.latest_complete_step()
            if latest is None:
                raise FileNotFoundError(
                    f"no complete checkpoint under {ckpt_parent}; "
                    f"expected step-XXXXX/manifest.json"
                )
            step = latest

    log.info(
        "load_checkpoint_for_inference_start",
        ckpt=str(ckpt_parent), step=step,
    )

    # Tokenizer.
    tok_path = ensure_tokenizer_local(str(tokenizer_path), tokenizer_key)
    tokenizer = load_tokenizer(tok_path)
    verify_tokenizer_has_required(tokenizer)
    pad_id = int(tokenizer.token_to_id(SpecialTokens.PAD))
    eos_id = int(tokenizer.token_to_id(SpecialTokens.EOS))

    # Model + optimizer (optimizer only used to build a state template
    # so Orbax can match the saved pytree structure; we never call
    # optimizer.update()).
    model_cfg = ModelConfig.from_yaml(str(model_config_path))
    opt_cfg = OptimizerConfig(
        peak_lr=3e-4, beta1=0.9, beta2=0.95, weight_decay=0.1, eps=1e-8,
    )
    model, optimizer = init_model_and_optimizer(
        model_cfg, opt_cfg, total_steps=max(1, step + 1),
    )
    state = initial_train_state(model, optimizer)
    template = {
        k: state[k] for k in (
            "trainable_variables", "non_trainable_variables", "opt_state",
            "step", "lr_recovery_multiplier", "data_position",
        ) if k in state
    }

    # Sharding for cross-mesh restore (G6 fix). Single-device inference
    # pods need SingleDeviceSharding; multi-device get NamedSharding
    # replicate. Required because checkpoints saved on N devices won't
    # restore on M devices without an explicit per-leaf sharding spec.
    devices = jax.devices()
    if len(devices) == 1:
        sharding = jax.sharding.SingleDeviceSharding(devices[0])
    else:
        mesh = jax.sharding.Mesh(devices, axis_names=("data",))
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    ckpt_mgr = CheckpointManager(
        CheckpointConfig(
            root=str(ckpt_parent), keep_last_n=1, keep_every_n=10**9,
            r2_prefix=None,
        )
    )
    restored = ckpt_mgr.restore(step, template=template, sharding=sharding)

    # JIT the forward — small wrapper around model.stateless_call so
    # caller doesn't have to remember the Keras call shape.
    def _forward(trainable, non_trainable, input_ids):
        logits, _ = model.stateless_call(trainable, non_trainable, input_ids)
        return logits

    forward_jit = jax.jit(_forward)

    log.info(
        "load_checkpoint_for_inference_done",
        step=step, n_devices=len(devices),
        sharding=type(sharding).__name__,
    )

    return LoadedCheckpoint(
        model=model,
        trainable=restored["trainable_variables"],
        non_trainable=restored["non_trainable_variables"],
        model_cfg=model_cfg,
        tokenizer=tokenizer,
        forward_jit=forward_jit,
        ctx_length=int(model_cfg.context_length),
        pad_id=pad_id,
        eos_id=eos_id,
        step=step,
    )


def _greedy_continuation(
    bundle: LoadedCheckpoint,
    prompt: str,
    *,
    max_new_tokens: int,
) -> str:
    """One greedy continuation. Argmax of next-token logits, stop on EOS
    or max_new_tokens. Returns ONLY the generated continuation (not the
    prompt) so benchmark scorers see what the model produced."""
    import jax.numpy as jnp

    encoded = bundle.tokenizer.encode(prompt)
    prompt_ids = list(encoded.ids if hasattr(encoded, "ids") else encoded)
    if len(prompt_ids) >= bundle.ctx_length:
        prompt_ids = prompt_ids[-bundle.ctx_length:]
        log.warning("prompt_truncated_to_ctx", ctx_length=bundle.ctx_length)

    buf = jnp.full((1, bundle.ctx_length), bundle.pad_id, dtype=jnp.int32)
    for i, t in enumerate(prompt_ids):
        buf = buf.at[0, i].set(int(t))

    position = len(prompt_ids)
    generated: list[int] = []

    for _ in range(max_new_tokens):
        if position >= bundle.ctx_length:
            break
        logits = bundle.forward_jit(bundle.trainable, bundle.non_trainable, buf)
        next_token = int(jnp.argmax(logits[0, position - 1, :]))
        if next_token == bundle.eos_id:
            break
        buf = buf.at[0, position].set(next_token)
        generated.append(next_token)
        position += 1

    return bundle.tokenizer.decode(generated)


def build_greedy_predict_fn(
    *,
    model_config_path: Path,
    tokenizer_path: Path,
    checkpoint_root: Path,
    checkpoint_step: int | None = None,
    max_new_tokens: int = 80,
    tokenizer_key: str | None = None,
) -> Callable[[str], str]:
    """Return a ``predict_fn(prompt: str) -> str`` for benchmark scoring.

    The returned callable does greedy decoding (no sampling). Each call
    produces up to ``max_new_tokens`` of continuation, stops on EOS.
    The model is loaded once at build time; subsequent calls reuse the
    on-device state.

    For sampling / top-p decoding use ``scripts/generate.py``'s
    ``generate_one`` directly — this is a deliberately simple path for
    the scorecard's "did the model produce a plausible answer?" check.
    """
    bundle = load_checkpoint_for_inference(
        model_config_path=Path(model_config_path),
        tokenizer_path=Path(tokenizer_path),
        checkpoint_root=Path(checkpoint_root),
        checkpoint_step=checkpoint_step,
        tokenizer_key=tokenizer_key,
    )

    def predict(prompt: str) -> str:
        return _greedy_continuation(bundle, prompt, max_new_tokens=max_new_tokens)

    return predict
