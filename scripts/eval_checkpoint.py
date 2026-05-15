#!/usr/bin/env python3
"""Standalone post-hoc validation-loss evaluator for a saved checkpoint.

Computes ``val_loss`` (mean cross-entropy) and ``val_ppl`` (exp(val_loss)) on
the first N batches of a packed corpus, mirroring the in-training eval hook's
"held-out = first N batches off the iterator" semantics.

Why this exists
---------------
The in-training eval hook (``src/myllm/training/eval_hook.py``) stopped firing
on the pilot-250m-v1 run after the crash-resume at step 65000: it reuses
``train_step_fn`` which was hit by an int32 overflow on ``data_position`` once
the cumulative token counter passed 2^31 (~2.1B tokens). The training loop's
direct call to ``train_step_fn`` was fixed in commit 9f442f7 (pop
``data_position`` before the JIT'd call, restore after), but the eval hook's
call still has the bug, so every post-resume eval logged
``eval_failed_non_fatal`` and produced no number.

This script gives us ONE number for the final checkpoint without re-launching
training:
  - Forward-only path — no optimizer, no opt_state, no gradient computation.
  - No ``data_position`` anywhere in the JIT'd code path, so the int32
    overflow that broke the in-training hook is impossible here.
  - The first ``--n-batches`` (default 32) batches of the packed corpus are
    used. These are exactly the batches the in-training eval hook held out
    from training (see run_pretrain.py:1310 ``take_held_out_batches``), so
    the model has not seen them during training. They were the same batches
    the step-65000 eval reported on (val_loss=2.9750), making this a direct
    apples-to-apples follow-up read.

Usage
-----
    python scripts/eval_checkpoint.py \\
        --checkpoint /workspace/ckpt/pilot-250m-v1/step-000151990 \\
        --model-config configs/pilot_250m.yaml \\
        --tokenizer-path artifacts/tokenizer.json \\
        --packed-corpus-root /workspace/data/pilot_corpus \\
        --n-batches 32 \\
        --output-json /workspace/ckpt/pilot-250m-v1/step-000151990/eval.json

Notes
-----
1. The checkpoint directory must be a ``step-NNNNNNNNN/`` directory with a
   completed ``manifest.json`` (the Orbax/CheckpointManager completion marker).
2. ``--micro-batch`` should match what the training run used so the batches
   line up sequence-for-sequence with what the in-training hook would have
   seen. The pilot used 4 (resolved from the model yaml).
3. ``--skip-batches`` lets you evaluate on a different slice (e.g. for a
   sanity check on a never-evaluated slice); the default 0 reproduces the
   in-training hook's "first N" held-out set exactly.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

# IMPORTANT: set Keras backend BEFORE importing keras anywhere.
os.environ.setdefault("KERAS_BACKEND", "jax")

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
# Also expose the repo root so ``from scripts.run_pretrain import ...``
# resolves (matches the pattern in scripts/benchmark_throughput.py).
sys.path.insert(0, str(_REPO))

# Reuse the production launcher's helpers so this script can't drift from
# the training run's config-loading and state-construction conventions.
from scripts.run_pretrain import (  # noqa: E402
    batch_pairs,
    ensure_tokenizer_local,
    init_model_and_optimizer,
    initial_train_state,
    load_yaml,
)

from myllm.data.packed_corpus import (  # noqa: E402
    PackedCorpusReader,
    iter_packed_pairs,
)
from myllm.data.special_tokens import SpecialTokens, verify_tokenizer_has_required  # noqa: E402
from myllm.data.tokenize import load_tokenizer  # noqa: E402
from myllm.model import ModelConfig  # noqa: E402
from myllm.training.checkpoint import CheckpointConfig, CheckpointManager  # noqa: E402
from myllm.training.loss import cross_entropy_with_z_loss  # noqa: E402
from myllm.training.optimizer import OptimizerConfig  # noqa: E402
from myllm.utils import configure_logging, get_logger  # noqa: E402

log = get_logger(__name__)


def _parse_step_from_path(checkpoint_dir: Path) -> int:
    """Pull the integer step out of a ``step-NNNNNNNNN`` directory name."""
    name = checkpoint_dir.name
    if not name.startswith("step-"):
        raise ValueError(
            f"--checkpoint must point to a 'step-NNNNNNNNN/' directory; "
            f"got {checkpoint_dir}"
        )
    try:
        return int(name.split("-", 1)[1])
    except ValueError as e:
        raise ValueError(
            f"could not parse step number from {checkpoint_dir.name!r}"
        ) from e


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a single 'step-NNNNNNNNN/' directory (the Orbax save "
             "dir for one training step). Must contain a manifest.json.",
    )
    p.add_argument(
        "--model-config",
        required=True,
        help="YAML model config (same one --run_pretrain.py was launched with).",
    )
    p.add_argument(
        "--tokenizer-path",
        default="artifacts/tokenizer.json",
        help="Local path to the BPE tokenizer JSON. Used only to look up the "
             "PAD/EOS token ids needed by the loss + segment masking.",
    )
    p.add_argument(
        "--packed-corpus-root",
        required=True,
        help="Path to the packed corpus root (the dir containing manifest.json "
             "and the shard-NNNNNN/ subdirs).",
    )
    p.add_argument(
        "--n-batches",
        type=int,
        default=32,
        help="Number of batches to evaluate on (default 32). The in-training "
             "eval hook used 8; we use 32 here for a tighter mean estimate "
             "since this is a one-shot post-hoc eval, not a per-step cost.",
    )
    p.add_argument(
        "--micro-batch",
        type=int,
        default=4,
        help="Per-device batch size. Default 4 matches the pilot-250m-v1 "
             "training run (resolved from configs/pilot_250m.yaml).",
    )
    p.add_argument(
        "--skip-batches",
        type=int,
        default=0,
        help="Skip this many batches before starting the eval slice. Default "
             "0 reproduces the in-training hook's 'first N held out off the "
             "top' semantics; set to a positive value if you want a different "
             "(never-seen-by-training) slice.",
    )
    p.add_argument(
        "--output-json",
        default=None,
        help="If set, write {val_loss, val_ppl, n_batches, checkpoint, "
             "skip_batches} to this path on completion.",
    )
    args = p.parse_args()

    configure_logging()

    # ---------------------------------------------------------------- #
    # 1. Parse checkpoint dir + step.
    # ---------------------------------------------------------------- #
    checkpoint_dir = Path(args.checkpoint).resolve()
    if not checkpoint_dir.exists():
        log.error("checkpoint_dir_missing", path=str(checkpoint_dir))
        return 2
    if not (checkpoint_dir / "manifest.json").exists():
        log.error(
            "checkpoint_manifest_missing",
            path=str(checkpoint_dir),
            msg="No manifest.json — checkpoint is incomplete or wrong dir.",
        )
        return 2
    step = _parse_step_from_path(checkpoint_dir)
    local_root = checkpoint_dir.parent
    log.info("eval_checkpoint_target", step=step, dir=str(checkpoint_dir))

    # ---------------------------------------------------------------- #
    # 2. Load configs (same shape as run_pretrain.main()).
    # ---------------------------------------------------------------- #
    model_cfg = ModelConfig.from_yaml(args.model_config)
    yaml_lr_schedule = load_yaml(args.model_config).get("lr_schedule", {})
    yaml_peak_lr = yaml_lr_schedule.get("peak_lr")
    peak_lr_value = float(yaml_peak_lr) if yaml_peak_lr is not None else 2.0e-4
    log.info(
        "model_config_loaded",
        name=model_cfg.name,
        params_estimate=model_cfg.param_count_estimate(),
        vocab_size=model_cfg.vocab_size,
        context_length=model_cfg.context_length,
        peak_lr=peak_lr_value,
    )

    # ---------------------------------------------------------------- #
    # 3. Tokenizer — need PAD id for loss masking + EOS for completeness.
    # ---------------------------------------------------------------- #
    tok_path = ensure_tokenizer_local(args.tokenizer_path, None)
    tokenizer = load_tokenizer(tok_path)
    verify_tokenizer_has_required(tokenizer)
    pad_id = tokenizer.token_to_id(SpecialTokens.PAD)

    # ---------------------------------------------------------------- #
    # 4. Build model + optimizer template. We need the optimizer ONLY
    #    so initial_train_state can construct the template pytree that
    #    Orbax restore matches against (preserves the
    #    optax.MultiTransformState namedtuple — see checkpoint.py:118).
    #    The optimizer is never used to update weights here.
    # ---------------------------------------------------------------- #
    # total_steps doesn't affect weight shapes — only the lr schedule's
    # length. Pick something nonzero so resolve_wsd_schedule_params doesn't
    # divide-by-zero; the schedule itself is irrelevant to eval.
    opt_cfg = OptimizerConfig(peak_lr=peak_lr_value)
    model, optimizer = init_model_and_optimizer(
        model_cfg,
        opt_cfg,
        total_steps=max(1, step + 1),
        lr_schedule_cfg=yaml_lr_schedule,
    )
    template_state = initial_train_state(model, optimizer)
    log.info(
        "model_built",
        n_trainable=len(model.trainable_variables),
        n_non_trainable=len(model.non_trainable_variables),
    )

    # ---------------------------------------------------------------- #
    # 5. Restore the checkpoint via CheckpointManager with the template.
    #    The template is critical so optax.MultiTransformState is rebuilt
    #    correctly (B1 fix in the 2026-05-12 audit). We don't USE opt_state
    #    here, but the template must match the saved structure or Orbax
    #    will surface a tree-shape mismatch.
    # ---------------------------------------------------------------- #
    ckpt_mgr = CheckpointManager(
        CheckpointConfig(root=str(local_root), keep_last_n=1, r2_prefix=None)
    )
    # G6 reshard: build sharding for current devices so we can restore
    # checkpoints saved on a different topology (e.g., DP=4 -> DP=1).
    devices = jax.devices()
    if len(devices) == 1:
        sharding = jax.sharding.SingleDeviceSharding(devices[0])
    else:
        mesh = jax.sharding.Mesh(devices, axis_names=("data",))
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    restored = ckpt_mgr.restore(step, template=template_state, sharding=sharding)
    trainable_vars = restored["trainable_variables"]
    non_trainable_vars = restored["non_trainable_variables"]
    log.info("checkpoint_restored_ok", step=step, n_devices=len(devices))

    # ---------------------------------------------------------------- #
    # 6. Build the batch iterator from the packed corpus. We mirror the
    #    training-time data path exactly:
    #      iter_packed_pairs(reader, start_sequence_id=0)
    #        -> batch_pairs(pair_iter, micro_batch, model_input_len)
    #    The first ``--skip-batches + --n-batches`` batches off this
    #    iterator match what the training run consumed (or held out)
    #    sequence-for-sequence.
    # ---------------------------------------------------------------- #
    expected_packed_seq_len = int(model_cfg.context_length) + 1
    reader = PackedCorpusReader(args.packed_corpus_root)
    if reader.sequence_length != expected_packed_seq_len:
        log.error(
            "packed_corpus_sequence_length_mismatch",
            corpus_seq_len=reader.sequence_length,
            expected=expected_packed_seq_len,
            model_context_length=model_cfg.context_length,
        )
        return 3
    model_input_len = int(model_cfg.context_length)
    log.info(
        "packed_corpus_opened",
        root=str(args.packed_corpus_root),
        total_sequences=reader.total_sequences,
        total_tokens=reader.total_tokens,
        sequence_length=reader.sequence_length,
        model_input_len=model_input_len,
    )

    pair_iter = iter_packed_pairs(reader, start_sequence_id=0)
    batch_iter = batch_pairs(pair_iter, args.micro_batch, model_input_len)

    # ---------------------------------------------------------------- #
    # 7. Forward-only eval step. NO data_position, NO opt_state — only
    #    (trainable, non_trainable, batch) -> scalar CE loss. This is
    #    the smallest possible JIT trace; the int32-overflow bug that
    #    broke the in-training hook can't reach this path because
    #    nothing here grows unboundedly with training step count.
    # ---------------------------------------------------------------- #
    import jax  # noqa: E402

    def _eval_loss(trainable, non_trainable, batch):
        segment_ids = batch.get("segment_ids")
        call_kwargs = {}
        if segment_ids is not None:
            call_kwargs["segment_ids"] = segment_ids
        logits, _ = model.stateless_call(
            trainable, non_trainable, batch["input_ids"], **call_kwargs,
        )
        loss, _metrics = cross_entropy_with_z_loss(
            logits,
            batch["labels"],
            ignore_index=pad_id,
            z_loss_coef=model_cfg.z_loss_coef,
            loss_mask=batch.get("loss_mask"),
        )
        return loss

    eval_step = jax.jit(_eval_loss)

    # ---------------------------------------------------------------- #
    # 8. Skip the warmup-skip batches (cheaply — we just consume them
    #    from the iterator), then take --n-batches and accumulate loss.
    # ---------------------------------------------------------------- #
    skipped = 0
    for _ in range(args.skip_batches):
        try:
            next(batch_iter)
            skipped += 1
        except StopIteration:
            log.error(
                "iterator_exhausted_during_skip",
                requested_skip=args.skip_batches,
                actually_skipped=skipped,
            )
            return 4
    if skipped:
        log.info("skipped_batches", n=skipped)

    losses: list[float] = []
    for batch_idx in range(args.n_batches):
        try:
            batch = next(batch_iter)
        except StopIteration:
            log.warning(
                "iterator_exhausted_during_eval",
                requested=args.n_batches,
                got=len(losses),
            )
            break
        loss = float(eval_step(trainable_vars, non_trainable_vars, batch))
        if math.isfinite(loss):
            losses.append(loss)
            log.info(
                "eval_batch",
                idx=batch_idx,
                loss=round(loss, 4),
            )
        else:
            log.warning("eval_batch_non_finite", idx=batch_idx, loss=loss)

    if not losses:
        log.error("eval_no_finite_loss", n_attempted=args.n_batches)
        return 5

    val_loss = sum(losses) / len(losses)
    try:
        val_ppl = float(math.exp(val_loss))
    except OverflowError:
        val_ppl = float("inf")

    # ---------------------------------------------------------------- #
    # 9. Report. Print to stdout + optionally write JSON.
    # ---------------------------------------------------------------- #
    print("=" * 60)
    print(f"checkpoint      : {checkpoint_dir}")
    print(f"step            : {step}")
    print(f"skip_batches    : {args.skip_batches}")
    print(f"n_batches       : {len(losses)} (of {args.n_batches} requested)")
    print(f"micro_batch     : {args.micro_batch}")
    print(f"val_loss        : {val_loss:.6f}")
    print(f"val_ppl         : {val_ppl:.4f}")
    print("=" * 60)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "val_loss": float(val_loss),
                    "val_ppl": float(val_ppl),
                    "n_batches": int(len(losses)),
                    "checkpoint": str(checkpoint_dir),
                    "skip_batches": int(args.skip_batches),
                },
                indent=2,
            )
            + "\n"
        )
        log.info("eval_json_written", path=str(out_path))

    return 0


if __name__ == "__main__":
    sys.exit(main())
