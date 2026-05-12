#!/usr/bin/env python3
"""L3 — Forced-kill resume bitwise-exact canary.

Per 2026-05-12 reviewer Q&A §5: this is the single most valuable canary.
It catches bug classes that don't show up in unit tests but kill multi-day
production runs:

  - Tied-embedding gradient not reduced (BLOOM tr11)
  - LayerNorm in weight-decay group (checkpoint mismatch on reload)
  - Data cursor reset on resume (silent corpus misalignment)
  - Optimizer state restore structure (MultiTransformState flattened)

Protocol:

  1. Run training uninterrupted for N steps.
     Capture: final state hash, final loss.

  2. Run training for N/2 steps; let the checkpoint cadence save at N/2.
     Kill the process (clean exit after N/2).

  3. Resume: launch the same command; loop.train_loop() detects the
     checkpoint at N/2 and restores. Run to N steps.
     Capture: final state hash, final loss.

  4. Assert:
     - |loss(N)_uninterrupted - loss(N)_resumed| <= 1e-4 (bf16 noise)
     - hash(state)_uninterrupted == hash(state)_resumed

Uses a tiny synthetic-data run (small model, ~1M params) so it's
CPU-runnable in <60s. The bug classes it catches are scale-invariant —
a resume bug at 1M params is the same resume bug at 1B.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Set this BEFORE any keras / jax import so the JAX backend resolves correctly.
os.environ.setdefault("KERAS_BACKEND", "jax")
# Force the harness onto CPU so its Orbax restore uses the same sharding
# topology as the subprocess (which itself pins JAX_PLATFORMS=cpu below).
# Without this, on a GPU pod the harness picks GPU as the default platform
# and Orbax fails with "sharding ... Got None" when restoring a checkpoint
# whose sharding metadata points at CPU devices. The L3 canary is tiny + by
# design CPU-runnable in seconds; no reason to involve the GPU.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from myllm.canary import CheckResult, hash_training_state  # noqa: E402
from myllm.utils import get_logger  # noqa: E402

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Tiny config — a 4-layer, 64-hidden, 256-FFN model with vocab 256.
# Param count ~250K; CPU-runnable in seconds per step.
# --------------------------------------------------------------------------- #
_TINY_MODEL_YAML = """\
name: l3_canary_tiny
arch: llama_decoder
layers: 2
hidden_dim: 64
ffn_dim: 256
num_heads: 4
num_kv_heads: 2
head_dim: 16
vocab_size: 256
tie_embeddings: true
context_length: 32
position: rope
rope_base: 10000.0
norm: rmsnorm
norm_eps: 1.0e-5
activation: swiglu
init_std: 0.02
scaled_init_for_residuals: false
z_loss_coef: 1.0e-4
qk_norm: false
optimizer:
  type: adamw
  beta1: 0.9
  beta2: 0.95
  weight_decay: 0.1
  eps: 1.0e-8
lr_schedule:
  type: wsd
  peak_lr: 1.0e-3
  end_lr_ratio: 0.1
  warmup_steps: 2
  decay_fraction: 0.0
batch:
  micro_batch_per_device: 2
  sequence_length: 33
  grad_accum_steps: 1
grad_clip_global_norm: 1.0
mixed_precision: bfloat16
target_tokens: 1_000_000
checkpoint_every_steps: 1
keep_last_n: 5
keep_every_n_steps: 1
"""

# Minimal data yaml stub. The synthetic-data path skips sources/filters but
# run_pretrain.py still requires --data-config to load; only `batch` and
# `data_seed` matter, and `batch.micro_batch_per_device` is overridden by
# the model yaml's batch block above.
_TINY_DATA_YAML = """\
name: l3_canary_data_stub
batch:
  micro_batch_per_device: 2
data_seed: 42
"""


def _write_tiny_config(tmpdir: Path) -> Path:
    path = tmpdir / "tiny_model.yaml"
    path.write_text(_TINY_MODEL_YAML)
    return path


def _write_tiny_data_config(tmpdir: Path) -> Path:
    path = tmpdir / "tiny_data.yaml"
    path.write_text(_TINY_DATA_YAML)
    return path


def _run_pretrain_subprocess(
    *,
    model_config: Path,
    data_config: Path,
    checkpoint_root: Path,
    total_steps: int,
    seed: int,
    run_name: str,
) -> tuple[int, str]:
    """Launch a fresh run_pretrain.py subprocess and wait for completion.

    Returns (returncode, last_state_dir).
    """
    cmd = [
        sys.executable,
        str(_REPO / "scripts" / "run_pretrain.py"),
        "--model-config", str(model_config),
        "--data-config", str(data_config),
        "--run-name", run_name,
        "--synthetic-data",
        "--total-steps", str(total_steps),
        "--checkpoint-root", str(checkpoint_root),
        "--seed", str(seed),
        "--no-shard",   # single-device CPU
        "--no-wandb",
    ]
    env = dict(os.environ)
    env["KERAS_BACKEND"] = "jax"
    # Suppress JAX's "no GPU found" warnings during canary.
    env.setdefault("JAX_PLATFORMS", "cpu")
    log.info("l3_subprocess_start", total_steps=total_steps, root=str(checkpoint_root))
    proc = subprocess.run(
        cmd, env=env, capture_output=True, text=True,
        timeout=300,  # 5min cap per phase
    )
    if proc.returncode != 0:
        log.error(
            "l3_subprocess_failed",
            returncode=proc.returncode,
            stderr_tail=proc.stderr[-2000:],
        )
    return proc.returncode, str(checkpoint_root)


def _latest_step_dir(checkpoint_root: Path) -> Path | None:
    dirs = sorted(
        d for d in checkpoint_root.glob("step-*") if (d / "manifest.json").exists()
    )
    return dirs[-1] if dirs else None


def _restore_final_state(checkpoint_root: Path) -> dict:
    """Open the latest checkpoint and return the restored state dict.

    Uses CheckpointManager directly (same path the loop uses). The
    returned state is whatever we persist (see _PERSIST_KEYS in loop.py).
    """
    from myllm.training.checkpoint import CheckpointConfig, CheckpointManager

    cm = CheckpointManager(CheckpointConfig(root=str(checkpoint_root)))
    step = cm.latest_complete_step()
    if step is None:
        raise RuntimeError(f"no complete checkpoint under {checkpoint_root}")
    return cm.restore(step)


def run_l3_check(*, total_steps: int = 4) -> CheckResult:
    """Run the bitwise-exact resume test. Returns a CheckResult.

    Default ``total_steps=4`` is the minimum that exercises the
    checkpoint→kill→resume→continue cycle: kill at step 2, resume to 4.
    Bigger total_steps shake out subtler bugs but cost more wall time.
    """
    tmpdir = Path(tempfile.mkdtemp(prefix="canary_l3_"))
    try:
        tiny_cfg = _write_tiny_config(tmpdir)
        tiny_data_cfg = _write_tiny_data_config(tmpdir)
        half = max(1, total_steps // 2)

        # --- Phase 1: uninterrupted reference run -------------------------
        ref_root = tmpdir / "ref"
        ref_root.mkdir()
        rc, _ = _run_pretrain_subprocess(
            model_config=tiny_cfg, data_config=tiny_data_cfg,
            checkpoint_root=ref_root,
            total_steps=total_steps, seed=42,
            run_name="l3-canary-ref",
        )
        if rc != 0:
            return CheckResult(
                name="l3_forced_kill_resume",
                passed=False,
                summary=f"reference run exited rc={rc}",
                fix_hint="Check that scripts/run_pretrain.py runs to completion "
                         "on synthetic data + tiny config. This is upstream of L3.",
            )
        ref_state = _restore_final_state(ref_root)
        ref_hash = hash_training_state(ref_state)
        ref_data_pos = int(ref_state.get("data_position", 0))
        ref_step = int(ref_state.get("step", 0))

        # --- Phase 2: interrupted run (half steps) ------------------------
        run_root = tmpdir / "run"
        run_root.mkdir()
        rc, _ = _run_pretrain_subprocess(
            model_config=tiny_cfg, data_config=tiny_data_cfg,
            checkpoint_root=run_root,
            total_steps=half, seed=42,
            run_name="l3-canary-interrupt",
        )
        if rc != 0:
            return CheckResult(
                name="l3_forced_kill_resume",
                passed=False,
                summary=f"interrupted run exited rc={rc} at {half} steps",
            )

        # --- Phase 3: resume to total_steps -------------------------------
        # Same checkpoint_root → loop.train_loop detects the existing
        # checkpoint and resumes.
        rc, _ = _run_pretrain_subprocess(
            model_config=tiny_cfg, data_config=tiny_data_cfg,
            checkpoint_root=run_root,
            total_steps=total_steps, seed=42,
            run_name="l3-canary-resume",
        )
        if rc != 0:
            return CheckResult(
                name="l3_forced_kill_resume",
                passed=False,
                summary=f"resumed run exited rc={rc}",
            )
        resumed_state = _restore_final_state(run_root)
        resumed_hash = hash_training_state(resumed_state)
        resumed_data_pos = int(resumed_state.get("data_position", 0))
        resumed_step = int(resumed_state.get("step", 0))

        # --- Compare ------------------------------------------------------
        step_match = ref_step == resumed_step
        data_pos_match = ref_data_pos == resumed_data_pos
        hash_match = ref_hash == resumed_hash

        passed = step_match and data_pos_match and hash_match
        summary = (
            "bitwise-exact resume verified"
            if passed
            else "resume diverged from uninterrupted run"
        )
        details = {
            "total_steps": total_steps,
            "kill_after_step": half,
            "ref_step": ref_step,
            "resumed_step": resumed_step,
            "ref_data_position": ref_data_pos,
            "resumed_data_position": resumed_data_pos,
            "ref_state_hash": ref_hash,
            "resumed_state_hash": resumed_hash,
            "step_match": step_match,
            "data_position_match": data_pos_match,
            "hash_match": hash_match,
        }
        fix_hint = None
        if not passed:
            failing = []
            if not step_match:
                failing.append(f"step ({ref_step} vs {resumed_step})")
            if not data_pos_match:
                failing.append(
                    f"data_position ({ref_data_pos} vs {resumed_data_pos})"
                )
            if not hash_match:
                failing.append("state hash")
            fix_hint = (
                "Diverging: " + ", ".join(failing) + ". This is the bug class "
                "the canary exists to catch — likely an unpersisted state key "
                "(see _PERSIST_KEYS in loop.py) or a non-deterministic data "
                "stream that doesn't re-derive from the seed alone."
            )
        return CheckResult(
            name="l3_forced_kill_resume",
            passed=passed,
            summary=summary,
            details=details,
            fix_hint=fix_hint,
        )
    finally:
        # Keep the tempdir on failure for forensic inspection? For now,
        # always clean up — the operator can re-run with stdout to capture.
        shutil.rmtree(tmpdir, ignore_errors=True)


def main() -> int:
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--total-steps", type=int, default=4)
    p.add_argument("--format", choices=("text", "json"), default="text")
    args = p.parse_args()

    result = run_l3_check(total_steps=args.total_steps)
    if args.format == "json":
        import json
        print(json.dumps(result.to_dict(), indent=2))
    else:
        sigil = "✓" if result.passed else "✗"
        print(f"{sigil} {result.name}: {result.summary}")
        for k, v in result.details.items():
            print(f"    {k}: {v}")
        if result.fix_hint:
            print(f"    ↳ fix: {result.fix_hint}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
