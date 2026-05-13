#!/usr/bin/env python3
"""L2 — FSDP vs DP-replicated loss-curve parity canary.

The senior reviewer (2026-05-12, second pass) flagged a "do-not-pass-go"
condition for FSDP: single-device-vs-N-device loss parity. This canary
exercises the production code path that the L2 spec wants:

  1. Build a tiny synthetic-data training run.
  2. Phase A: run with DP-replicated state (the pre-FSDP default).
     Capture (step, loss) per step.
  3. Phase B: SAME tiny config, SAME seed, SAME batch — but with --fsdp.
     Capture (step, loss) per step.
  4. Assert: per-step loss values agree within 5e-3 (the agent plan's
     L2 tolerance; cross-device collectives reorder reductions, so
     bitwise equality is too strict).

Each phase is a fresh ``run_pretrain.py`` subprocess. We run both phases
on the SAME simulated multi-device CPU mesh via
``XLA_FLAGS=--xla_force_host_platform_device_count=4`` so the comparison
is between two different sharding strategies on identical hardware.

What this catches (the bug classes the agent flagged for FSDP):
  - "Silent grad replication" — if XLA emits all-reduce instead of
    reduce-scatter, the LOSS is still correct but bitwise differs
    enough across the run that the curves diverge.
  - donate_argnums + NaN-revert interaction breaks param update.
  - muP MultiTransformState flattening into a dict mid-update.
  - Sharding-spec mismatches that cause silent buffer-shape drift.

What this does NOT catch (saved for later):
  - Real GPU collective performance (CPU subprocess shim is functional
    not performance).
  - Bitwise determinism (we explicitly allow 5e-3 numerical slack).
  - Multi-host / multi-pod NCCL semantics.

Exit code 0 on PASS, 1 on FAIL (matches the canary_l3_resume.py convention).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Force JAX backend + 4 CPU "devices" before anything else.
os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault(
    "XLA_FLAGS", "--xla_force_host_platform_device_count=4"
)

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from myllm.canary import CheckResult  # noqa: E402
from myllm.utils import get_logger  # noqa: E402

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Tiny configs (mirror canary_l3_resume.py's pattern)
# --------------------------------------------------------------------------- #
_TINY_MODEL_YAML = """\
name: l2_canary_tiny
arch: llama_decoder
layers: 2
hidden_dim: 64
ffn_dim: 256
num_heads: 4
num_kv_heads: 2
head_dim: 16
vocab_size: 64
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
gradient_checkpointing: false
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
  # mb=4 = 1 sequence per device on a 4-device mesh
  micro_batch_per_device: 4
  sequence_length: 33
  grad_accum_steps: 1
grad_clip_global_norm: 1.0
mixed_precision: bfloat16
target_tokens: 1_000_000
checkpoint_every_steps: 1000
keep_last_n: 1
keep_every_n_steps: 1
"""

_TINY_DATA_YAML = """\
name: l2_canary_data_stub
batch:
  micro_batch_per_device: 4
data_seed: 42
"""


def _write_yaml(tmp: Path, name: str, content: str) -> Path:
    p = tmp / name
    p.write_text(content)
    return p


# --------------------------------------------------------------------------- #
# Subprocess driver
# --------------------------------------------------------------------------- #
def _run_pretrain(
    *,
    model_config: Path,
    data_config: Path,
    checkpoint_root: Path,
    total_steps: int,
    seed: int,
    run_name: str,
    fsdp: bool,
) -> tuple[int, list[tuple[int, float]]]:
    """Launch run_pretrain.py and parse per-step (step, loss) from stdout.

    Returns ``(returncode, [(step, loss), ...])``.
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
        "--no-wandb",
        "--no-watchdog",   # avoid spike-recovery side effects in the canary
        "--log-every", "1",  # need per-step events for the loss curve
    ]
    if fsdp:
        cmd.append("--fsdp")

    env = dict(os.environ)
    env["KERAS_BACKEND"] = "jax"
    env.setdefault("JAX_PLATFORMS", "cpu")
    env.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

    log.info(
        "l2_subprocess_start",
        fsdp=fsdp,
        total_steps=total_steps,
        root=str(checkpoint_root),
    )
    # IMPORTANT: structlog emits its JSON events to STDERR by default.
    # Redirect stderr -> stdout so we see them in `proc.stdout` for
    # parsing. Without this, the canary silently passes because the
    # captured stdout is empty and `max_abs_delta` stays at 0.0.
    proc = subprocess.run(
        cmd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, timeout=600,
    )
    if proc.returncode != 0:
        log.error(
            "l2_subprocess_failed",
            fsdp=fsdp,
            returncode=proc.returncode,
            stderr_tail=proc.stdout[-2000:],  # was merged above
        )
        return proc.returncode, []

    # Parse per-step events out of the JSONL log. structlog emits each
    # event on its own line as JSON; `event` field is "step" for the
    # per-step training log.
    curve: list[tuple[int, float]] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("event") != "step":
            continue
        try:
            step = int(ev["step"])
            loss = float(ev["loss"])
        except (KeyError, ValueError, TypeError):
            continue
        curve.append((step, loss))

    return proc.returncode, curve


# --------------------------------------------------------------------------- #
# Main check
# --------------------------------------------------------------------------- #
def run_l2_parity_check(*, total_steps: int = 5, atol: float = 5e-3) -> CheckResult:
    tmpdir = Path(tempfile.mkdtemp(prefix="canary_l2_"))
    try:
        model_cfg = _write_yaml(tmpdir, "tiny_model.yaml", _TINY_MODEL_YAML)
        data_cfg = _write_yaml(tmpdir, "tiny_data.yaml", _TINY_DATA_YAML)

        # Phase A — no-FSDP (DP-replicated)
        a_root = tmpdir / "no_fsdp"
        a_root.mkdir()
        rc_a, curve_a = _run_pretrain(
            model_config=model_cfg, data_config=data_cfg,
            checkpoint_root=a_root, total_steps=total_steps, seed=42,
            run_name="l2-canary-no-fsdp", fsdp=False,
        )
        if rc_a != 0:
            return CheckResult(
                name="l2_fsdp_parity",
                passed=False,
                summary=f"no-FSDP phase exited rc={rc_a}",
                fix_hint="Phase A (DP-replicated) failed before FSDP could "
                         "be compared. This is upstream of L2; check the "
                         "pre-FSDP run_pretrain path first.",
            )

        # Phase B — --fsdp
        b_root = tmpdir / "fsdp"
        b_root.mkdir()
        rc_b, curve_b = _run_pretrain(
            model_config=model_cfg, data_config=data_cfg,
            checkpoint_root=b_root, total_steps=total_steps, seed=42,
            run_name="l2-canary-fsdp", fsdp=True,
        )
        if rc_b != 0:
            return CheckResult(
                name="l2_fsdp_parity",
                passed=False,
                summary=f"FSDP phase exited rc={rc_b}",
                fix_hint="Phase B (--fsdp) failed. Most likely a sharding "
                         "mismatch in the run_pretrain init flow OR an "
                         "in_shardings/donate_argnums conflict in train_step. "
                         "See subprocess stderr in the failure event above.",
            )

        # Compare — both curves should have the SAME steps with losses
        # within `atol`.
        #
        # CRITICAL: empty curves => false positive. If we don't emit any
        # step events, max_abs_delta stays 0.0 and the test trivially
        # "passes" while validating nothing. Require at least one event
        # per phase.
        if not curve_a or not curve_b:
            return CheckResult(
                name="l2_fsdp_parity",
                passed=False,
                summary=(
                    f"no step events captured (no-FSDP={len(curve_a)}, "
                    f"FSDP={len(curve_b)}) — refusing to false-pass. Check "
                    f"that --log-every is forwarded to run_pretrain and "
                    f"that structlog output is being captured (stderr "
                    f"redirected into stdout)."
                ),
            )
        if len(curve_a) != len(curve_b):
            return CheckResult(
                name="l2_fsdp_parity",
                passed=False,
                summary=(
                    f"step count mismatch: no-FSDP emitted {len(curve_a)} "
                    f"step events, FSDP emitted {len(curve_b)}"
                ),
                details={
                    "no_fsdp_steps": [s for s, _ in curve_a],
                    "fsdp_steps": [s for s, _ in curve_b],
                },
            )

        max_abs_delta = 0.0
        max_delta_step = -1
        deltas: list[tuple[int, float, float, float]] = []
        for (sa, la), (sb, lb) in zip(curve_a, curve_b, strict=True):
            if sa != sb:
                return CheckResult(
                    name="l2_fsdp_parity",
                    passed=False,
                    summary=f"step alignment broken: no-FSDP step {sa} != FSDP step {sb}",
                )
            d = abs(la - lb)
            deltas.append((sa, la, lb, d))
            if d > max_abs_delta:
                max_abs_delta = d
                max_delta_step = sa

        passed = max_abs_delta <= atol
        summary = (
            f"FSDP vs DP-replicated loss curves agree within atol={atol} "
            f"(max |Δ|={max_abs_delta:.2e} at step {max_delta_step})"
            if passed
            else f"FSDP diverged from DP-replicated: max |Δ|={max_abs_delta:.2e} "
                 f"at step {max_delta_step} (atol={atol})"
        )
        fix_hint = None
        if not passed:
            fix_hint = (
                "Loss curves diverged. Likely root causes:\n"
                "  1. with_sharding_constraint on grads is missing or "
                "wrong (XLA falls back to all-reduce on a replicated "
                "grad tree -> different summation order -> drift). "
                "Check src/myllm/training/train_step.py:_train_step_body.\n"
                "  2. donate_argnums=(0,) interacting badly with the "
                "jnp.where NaN-revert (silent buffer aliasing). Try "
                "removing donate_argnums to bisect.\n"
                "  3. muP MultiTransformState flattened into a dict in the "
                "FSDP path — opt-state .inner_states becomes a key access "
                "and produces different updates. Walk shardings recursively "
                "and assert type identity at every level.\n"
                "  4. Param sharding axis differs across the run (e.g. "
                "ShapeDtypeStruct path vs real-array path picks different "
                "axes). Diff the param_shardings pytree between phases."
            )

        return CheckResult(
            name="l2_fsdp_parity",
            passed=passed,
            summary=summary,
            details={
                "total_steps": total_steps,
                "atol": atol,
                "max_abs_delta": max_abs_delta,
                "max_delta_step": max_delta_step,
                "loss_curves": [
                    {"step": s, "no_fsdp_loss": la, "fsdp_loss": lb, "delta": d}
                    for (s, la, lb, d) in deltas
                ],
            },
            fix_hint=fix_hint,
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main() -> int:
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--total-steps", type=int, default=5,
                   help="Steps per phase. Default 5 — enough to see drift "
                        "without paying full warmup. Bump to 20-50 for "
                        "tighter pre-launch validation.")
    p.add_argument("--atol", type=float, default=5e-3,
                   help="Per-step loss tolerance. Default 5e-3 per the "
                        "agent plan's L2 spec. Cross-device collectives "
                        "reorder reductions; bitwise equality is unrealistic.")
    p.add_argument("--format", choices=("text", "json"), default="text")
    args = p.parse_args()

    result = run_l2_parity_check(
        total_steps=args.total_steps, atol=args.atol,
    )
    if args.format == "json":
        print(json.dumps(result.to_dict(), indent=2))
    else:
        sigil = "✓" if result.passed else "✗"
        print(f"{sigil} {result.name}: {result.summary}")
        for k, v in result.details.items():
            if k == "loss_curves":
                # Print compact curve table
                print(f"    {k}:")
                for row in v:
                    print(
                        f"      step={row['step']:>3} "
                        f"no_fsdp={row['no_fsdp_loss']:.6f} "
                        f"fsdp={row['fsdp_loss']:.6f} "
                        f"Δ={row['delta']:.2e}"
                    )
            else:
                print(f"    {k}: {v}")
        if result.fix_hint:
            print(f"    ↳ fix:\n{result.fix_hint}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
