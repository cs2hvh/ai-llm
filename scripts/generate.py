#!/usr/bin/env python3
"""Generate text from a saved checkpoint — interactive smoke test.

Loads a checkpoint, runs autoregressive generation against prompts.
Useful for "is the model producing coherent text?" verification.

This is a BASE MODEL — it continues text, doesn't follow instructions.
Don't expect ChatGPT-style behavior. Expected behavior is on-style
continuation: "The capital of France is" → " Paris and it has..."
NOT "Answer: Paris."

Usage:
    # Single prompt
    python scripts/generate.py \\
        --checkpoint /workspace/ckpt/pilot-250m-v1-decay/step-000171990 \\
        --model-config configs/pilot_250m_decay.yaml \\
        --tokenizer-path artifacts/tokenizer_v1.json \\
        --prompt "The capital of France is" \\
        --max-new-tokens 50

    # Multiple prompts from a file
    python scripts/generate.py ... --prompts-file prompts.txt

    # Default smoke-test prompts (knowledge, code, multilingual)
    python scripts/generate.py ... (no --prompt or --prompts-file)

    # Greedy sampling (deterministic, picks highest-prob token each step)
    python scripts/generate.py ... --greedy

Throughput: ~50-100ms per generated token on H100 for a 250M model at
ctx=8192 (uses JIT-compiled forward over a padded ctx-sized buffer).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Backend setup MUST come before any jax/keras import
os.environ.setdefault("KERAS_BACKEND", "jax")

# Make repo + src/ importable
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "src"))

import jax
import jax.numpy as jnp

from myllm.model.config import ModelConfig
from myllm.training.checkpoint import CheckpointConfig, CheckpointManager
from myllm.training.optimizer import OptimizerConfig
from myllm.data.tokenize import load_tokenizer
from myllm.data.special_tokens import SpecialTokens
from myllm.utils.logging import configure_logging, get_logger

from scripts.run_pretrain import (
    init_model_and_optimizer,
    initial_train_state,
    ensure_tokenizer_local,
    load_yaml,
)

log = get_logger(__name__)


DEFAULT_PROMPTS = [
    "The capital of France is",
    "Two plus two equals",
    "Once upon a time, there was a",
    "def fibonacci(n):\n    ",
    "The Pacific Ocean is",
    "नमस्ते",
    "import numpy as np\ndef matmul(a, b):\n    ",
    "The President said",
    "In a recent scientific study,",
    "Theorem (Pythagorean):",
]


def _step_number_from_path(path: Path) -> int:
    """Extract the integer step number from a path like '.../step-000171990'."""
    name = path.name
    if not name.startswith("step-"):
        raise ValueError(f"Expected path ending in 'step-XXXXX', got {path}")
    return int(name[len("step-"):])


def sample_top_p(rng, logits, temperature: float, top_p: float):
    """Sample one token from logits using temperature + top-p (nucleus) sampling.

    Returns a scalar int jax array.
    """
    logits = logits / jnp.maximum(temperature, 1e-8)
    probs = jax.nn.softmax(logits, axis=-1)

    sorted_indices = jnp.argsort(-probs)
    sorted_probs = probs[sorted_indices]
    cumprobs = jnp.cumsum(sorted_probs)

    # Keep tokens whose cumulative prob is <= top_p, always keep top-1
    keep = cumprobs <= top_p
    keep = keep.at[0].set(True)

    filtered = jnp.where(keep, sorted_probs, 0.0)
    filtered = filtered / (filtered.sum() + 1e-12)

    chosen_local = jax.random.categorical(rng, jnp.log(filtered + 1e-30))
    return sorted_indices[chosen_local]


def build_and_restore(checkpoint_path: Path, model_config_path: Path):
    """Build the model from config, restore weights from checkpoint.

    Returns: (model, trainable_vars, non_trainable_vars, model_cfg)
    """
    log.info("loading_model_config", path=str(model_config_path))
    model_cfg = ModelConfig.from_yaml(model_config_path)

    # Optimizer is needed only to construct the state template (so Orbax
    # restore matches the saved pytree structure). We never call optimizer.update().
    opt_cfg = OptimizerConfig(
        peak_lr=3e-4,
        beta1=0.9, beta2=0.95,
        weight_decay=0.1,
        eps=1e-8,
    )

    step = _step_number_from_path(checkpoint_path)
    log.info("building_model_and_optimizer_template", step=step)
    model, optimizer = init_model_and_optimizer(
        model_cfg, opt_cfg, total_steps=max(1, step + 1)
    )

    state = initial_train_state(model, optimizer)
    template = {k: state[k] for k in (
        "trainable_variables", "non_trainable_variables", "opt_state",
        "step", "lr_recovery_multiplier", "data_position",
    ) if k in state}

    ckpt_cfg = CheckpointConfig(
        root=checkpoint_path.parent,
        keep_last_n=1,
        keep_every_n=10000,
        r2_prefix=None,
    )
    ckpt_mgr = CheckpointManager(ckpt_cfg)
    log.info("restoring_checkpoint", path=str(checkpoint_path))
    restored = ckpt_mgr.restore(step, template=template)

    return (
        model,
        restored["trainable_variables"],
        restored["non_trainable_variables"],
        model_cfg,
    )


def generate_one(
    *,
    model,
    trainable,
    non_trainable,
    tokenizer,
    forward_jit,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    greedy: bool,
    ctx_length: int,
    pad_id: int,
    eos_id: int,
    rng: jax.Array,
) -> tuple[str, jax.Array]:
    """Generate a continuation for one prompt. Returns (decoded_text, updated_rng)."""
    encoded = tokenizer.encode(prompt)
    prompt_ids = list(encoded.ids if hasattr(encoded, "ids") else encoded)

    if len(prompt_ids) >= ctx_length:
        prompt_ids = prompt_ids[-ctx_length:]
        log.warning("prompt_truncated_to_ctx", ctx_length=ctx_length)

    # Padded buffer of fixed ctx-length for JIT-stable shape
    buf = jnp.full((1, ctx_length), pad_id, dtype=jnp.int32)
    for i, t in enumerate(prompt_ids):
        buf = buf.at[0, i].set(int(t))

    position = len(prompt_ids)
    generated_ids: list[int] = list(prompt_ids)

    for _ in range(max_new_tokens):
        if position >= ctx_length:
            break

        # Forward pass — model sees the full padded buffer.
        # We read logits at position-1 (the last filled position predicts position).
        logits = forward_jit(trainable, non_trainable, buf)
        last_logits = logits[0, position - 1, :]

        if greedy:
            next_token = int(jnp.argmax(last_logits))
        else:
            rng, sub_rng = jax.random.split(rng)
            next_token = int(sample_top_p(sub_rng, last_logits, temperature, top_p))

        if next_token == eos_id:
            break

        buf = buf.at[0, position].set(next_token)
        generated_ids.append(next_token)
        position += 1

    decoded = tokenizer.decode(generated_ids)
    return decoded, rng


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="Path to step-XXXXXX directory")
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument("--tokenizer-path", default="artifacts/tokenizer_v1.json", type=Path)
    parser.add_argument("--tokenizer-key", default=None,
                        help="R2 key for tokenizer; used if --tokenizer-path doesn't exist locally")

    # Prompts: --prompt, --prompts-file, or default smoke-test set
    parser.add_argument("--prompt", type=str, default=None,
                        help="A single prompt to generate from")
    parser.add_argument("--prompts-file", type=Path, default=None,
                        help="File with one prompt per line")

    # Sampling
    parser.add_argument("--max-new-tokens", type=int, default=80,
                        help="Max tokens to generate per prompt (default 80)")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--greedy", action="store_true",
                        help="Deterministic greedy sampling (overrides temperature + top-p)")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    configure_logging()

    # Restore model
    model, trainable, non_trainable, model_cfg = build_and_restore(
        args.checkpoint, args.model_config
    )

    # Tokenizer
    tok_path = ensure_tokenizer_local(str(args.tokenizer_path), args.tokenizer_key)
    tokenizer = load_tokenizer(tok_path)
    pad_id = tokenizer.token_to_id(SpecialTokens.PAD)
    eos_id = tokenizer.token_to_id(SpecialTokens.EOS)
    if pad_id is None or eos_id is None:
        raise RuntimeError(
            f"Tokenizer missing PAD or EOS token: pad_id={pad_id}, eos_id={eos_id}"
        )
    log.info("tokenizer_loaded", pad_id=pad_id, eos_id=eos_id, vocab=tokenizer.get_vocab_size())

    # JIT the model forward over a fixed [1, ctx_length] input.
    # First call compiles; subsequent calls are fast (~50-100ms on H100 for 250M).
    ctx = model_cfg.context_length

    @jax.jit
    def forward_jit(trainable, non_trainable, input_ids):
        logits, _ = model.stateless_call(trainable, non_trainable, input_ids)
        return logits

    # Warm up JIT once with a dummy input
    log.info("compiling_forward", ctx_length=ctx)
    _ = forward_jit(
        trainable,
        non_trainable,
        jnp.full((1, ctx), pad_id, dtype=jnp.int32),
    )
    log.info("forward_compiled")

    # Get prompts
    if args.prompt:
        prompts = [args.prompt]
    elif args.prompts_file:
        prompts = [
            p.strip() for p in args.prompts_file.read_text().splitlines() if p.strip()
        ]
    else:
        prompts = DEFAULT_PROMPTS

    log.info("generation_start", n_prompts=len(prompts), max_new_tokens=args.max_new_tokens,
             greedy=args.greedy, temperature=args.temperature, top_p=args.top_p)

    rng = jax.random.PRNGKey(args.seed)

    for i, prompt in enumerate(prompts, start=1):
        print("=" * 72)
        print(f"[{i}/{len(prompts)}] PROMPT: {prompt!r}")
        print("-" * 72)
        text, rng = generate_one(
            model=model,
            trainable=trainable,
            non_trainable=non_trainable,
            tokenizer=tokenizer,
            forward_jit=forward_jit,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            greedy=args.greedy,
            ctx_length=ctx,
            pad_id=pad_id,
            eos_id=eos_id,
            rng=rng,
        )
        print(text)
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
