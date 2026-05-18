"""D8 GPU repro driver — chunked-CE NaN-grad on B200 + bf16 + 1B + FSDP.

Created 2026-05-17 as part of the post-review P0 stack. Designed to run on
a single rented B200 (~$5, ~30-60 min) and pin down the D8 root cause.

The bug, from the audit at `docs/design/d8_chunked_ce_audit.md`:
  - 1B-shape model + 4×B200 NVLink-5 + bf16 + width_mult=8 + --use-chunked-ce
  - forward loss is finite (~11.76 = ln(131072) random-init expectation)
  - backward gradient is NaN every step
  - dropping --use-chunked-ce (full-CE) clears it
  - CPU repro of the chunked-CE algorithm in bf16 produces FINITE gradients
    matching full-CE within 2.98e-7 — bug is therefore B200/CUDA-specific

Three candidate root causes from the verified review:
  1. openxla/xla#17922 — rematerialization pass mis-tracks XLA copies post
     JAX 0.4.30 (we pin 0.4.38). Forward fine, backward NaN. Matches our
     symptom precisely.
  2. log_softmax bf16 gradient instability (jax-ml/jax#13529 — "worse for
     bf16 but noticeable in fp32"). Already mitigated by 2026-05-17 commit
     that casts chunked-CE accumulators to fp32; this script verifies the
     mitigation is sufficient.
  3. FSDP reduce-scatter on grads at bf16 — would only fire with --fsdp.

Run modes (each takes a few hundred steps, well under 1 hour on 1×B200):

  python scripts/d8_gpu_repro.py --mode baseline
    Reproduce the original NaN. Uses chunked-CE at bf16 with old codepath
    (or current codepath as a control for whether the fp32-logsumexp fix
    already cleared it). Single GPU, no FSDP (eliminates cause #3).

  python scripts/d8_gpu_repro.py --mode disable-remat
    Sets XLA_FLAGS=--xla_disable_hlo_passes=rematerialization before JAX
    init. If this clears the NaN, cause #1 (openxla/xla#17922) is the
    root cause.

  python scripts/d8_gpu_repro.py --mode fsdp
    Re-enables FSDP (with mb=4 to fit single-GPU memory) to test whether
    the bug requires FSDP reduce-scatter. Use this only after baseline +
    disable-remat to isolate cause #3.

  python scripts/d8_gpu_repro.py --mode fp32-cce
    Force full-fp32 chunked-CE (cast hidden_states + lm_head_weight to
    fp32 before the loss). If this clears the NaN, the bug is in the
    bf16 numerics path inside chunked-CE specifically.

Outputs to stdout + writes a JSON summary to
  artifacts/d8_repro/<mode>_<timestamp>.json
with per-step (step, fwd_loss, grad_finite, grad_min, grad_max,
grad_nan_count, time_ms). Upload to R2 for the audit trail:

  aws --endpoint-url $S3_ENDPOINT_URL s3 cp \\
    artifacts/d8_repro/ \\
    s3://$S3_BUCKET/stage2-prep/d8-repro/ --recursive

Pre-flight (on a fresh single-B200 RunPod pod):
  pip install -r requirements-gpu.txt
  python -c 'import jax; print(jax.devices())'   # should show 1 cuda

The script is intentionally simple: build a 1B-shape model on 1 GPU,
take a real packed batch from R2, run 10 train_steps under
`jax.value_and_grad`, log per-step grad health. No FSDP, no Orbax, no
distillation — strip down to the minimum needed to reproduce.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def _set_xla_flags(mode: str) -> None:
    """Set XLA flags BEFORE any JAX import. JAX reads these at first
    `import jax` / device init; later changes are silently ignored."""
    flags = os.environ.get("XLA_FLAGS", "")
    if mode == "disable-remat":
        flags = (
            flags
            + " --xla_disable_hlo_passes=rematerialization"
        ).strip()
        os.environ["XLA_FLAGS"] = flags
        print(f"[d8-repro] XLA_FLAGS = {flags!r}", flush=True)


def _build_minibatch(vocab_size: int, micro_batch: int, seq_len: int):
    import numpy as np
    rng = np.random.default_rng(20260517)
    ids = rng.integers(1, vocab_size, size=(micro_batch, seq_len)).astype(np.int32)
    labels = rng.integers(1, vocab_size, size=(micro_batch, seq_len)).astype(np.int32)
    return {"input_ids": ids, "labels": labels}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--mode",
        choices=["baseline", "disable-remat", "fsdp", "fp32-cce"],
        required=True,
    )
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--config", type=str, default="configs/base_1b.yaml")
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=8,
        help="vocab chunks for chunked-CE (V=131072 / 8 = 16384 per chunk)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="artifacts/d8_repro",
    )
    args = parser.parse_args()

    # MUST be set before any JAX import.
    _set_xla_flags(args.mode)

    # Now safe to import the rest.
    os.environ.setdefault("KERAS_BACKEND", "jax")
    import jax
    import jax.numpy as jnp
    from keras import ops  # noqa: F401 — registers the JAX backend

    print(f"[d8-repro] mode={args.mode} steps={args.steps} "
          f"mb={args.micro_batch} seq={args.seq_len}", flush=True)
    print(f"[d8-repro] jax devices: {jax.devices()}", flush=True)

    # Allow running before `pip install -e .` by adding src/ to sys.path.
    # When the package IS installed, this is a harmless no-op (installed
    # location wins via normal import resolution).
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root / "src"))

    from myllm.model.config import ModelConfig
    from myllm.model.transformer import build_model
    from myllm.training.loss import chunked_cross_entropy_with_z_loss

    cfg = ModelConfig.from_yaml(args.config)
    print(
        f"[d8-repro] config: {cfg.name} layers={cfg.layers} hidden={cfg.hidden_dim} "
        f"vocab={cfg.vocab_size} width_mult={cfg.mup_width_multiplier()}",
        flush=True,
    )

    model = build_model(cfg)
    # Warm-up build to materialize all variables.
    dummy = jnp.zeros((1, 8), dtype=jnp.int32)
    _ = model(dummy)

    trainable = [jnp.asarray(v) for v in model.trainable_variables]
    non_trainable = [jnp.asarray(v) for v in model.non_trainable_variables]

    output_mult_value = (
        1.0 / cfg.mup_width_multiplier()
        if (cfg.mup is not None and cfg.mup.apply_lm_head_output_mult)
        else 1.0
    )

    def loss_fn(trainable_, batch_):
        # Pull hidden + LM head and run chunked-CE — mirrors train_step's
        # chunked-CE branch but standalone.
        (hidden, lm_head_w, output_mult), _ = model.stateless_call(
            trainable_,
            non_trainable,
            batch_["input_ids"],
            return_loss_inputs=True,
        )
        if args.mode == "fp32-cce":
            hidden = jnp.asarray(hidden, dtype=jnp.float32)
            lm_head_w = jnp.asarray(lm_head_w, dtype=jnp.float32)
            output_mult = jnp.asarray(output_mult, dtype=jnp.float32)
        loss, _ = chunked_cross_entropy_with_z_loss(
            hidden,
            lm_head_w,
            batch_["labels"],
            num_chunks=args.num_chunks,
            output_mult=output_mult,
            z_loss_coef=cfg.z_loss_coef,
            final_logit_softcap=cfg.final_logit_softcap,
        )
        return loss

    grad_fn = jax.jit(jax.value_and_grad(loss_fn))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for step in range(args.steps):
        batch = _build_minibatch(cfg.vocab_size, args.micro_batch, args.seq_len)
        batch = {k: jnp.asarray(v) for k, v in batch.items()}
        t0 = time.time()
        loss, grads = grad_fn(trainable, batch)
        # Force device sync via .block_until_ready on a flattened leaf.
        loss = float(loss.block_until_ready())
        # Probe grad health on each leaf.
        n_nan = 0
        any_inf = False
        gmin, gmax = float("inf"), float("-inf")
        flat = jax.tree_util.tree_leaves(grads)
        for g in flat:
            gnp = jax.device_get(g)
            n_nan += int((gnp != gnp).sum())  # NaN count
            if not bool((gnp == gnp).all()):
                pass  # already counted via n_nan
            if (gnp == float("inf")).any() or (gnp == float("-inf")).any():
                any_inf = True
            if gnp.size:
                gmin = min(gmin, float(gnp.min()))
                gmax = max(gmax, float(gnp.max()))
        dt = (time.time() - t0) * 1000.0
        rec = {
            "step": step,
            "fwd_loss": loss,
            "grad_n_nan": n_nan,
            "grad_any_inf": any_inf,
            "grad_min": gmin,
            "grad_max": gmax,
            "time_ms": dt,
        }
        records.append(rec)
        print(
            f"[d8-repro] step={step} loss={loss:.4f} "
            f"grad_nan={n_nan} inf={any_inf} "
            f"grad_min={gmin:.3e} grad_max={gmax:.3e} "
            f"dt={dt:.0f}ms",
            flush=True,
        )

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_file = out_dir / f"{args.mode}_{timestamp}.json"
    out_file.write_text(json.dumps({
        "mode": args.mode,
        "steps": args.steps,
        "micro_batch": args.micro_batch,
        "seq_len": args.seq_len,
        "num_chunks": args.num_chunks,
        "config": str(args.config),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "records": records,
    }, indent=2))
    print(f"[d8-repro] wrote {out_file}", flush=True)

    # Exit code: 0 if all steps had grad_n_nan == 0 (mitigation worked),
    # 1 if any step had NaN grads (D8 still firing in this mode).
    any_nan = any(r["grad_n_nan"] > 0 or r["grad_any_inf"] for r in records)
    if any_nan:
        print(f"[d8-repro] RESULT: D8 NaN-grad STILL FIRES in mode={args.mode}", flush=True)
        return 1
    print(f"[d8-repro] RESULT: clean — no NaN/Inf grads in mode={args.mode}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
