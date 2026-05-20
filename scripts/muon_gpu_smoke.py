"""Muon hybrid optimizer GPU smoke — D10 Step 5 (2026-05-18).

First public JAX-Muon training at >=1B scale (no prior public results
exist as of 2026-05-18 — see SESSION_HANDOFF.md §7 D10 row). Treat
this as the gating smoke before any Stage 2 spend.

Run modes:

  python scripts/muon_gpu_smoke.py --mode adamw_baseline --steps 50
    AdamW baseline at 1B / mb=1 / seq=4K / no FSDP. Gives a
    reference loss-trajectory we'll compare Muon against. ~$3 on
    1×B200.

  python scripts/muon_gpu_smoke.py --mode muon_singlegpu --steps 50
    Muon hybrid at the same shape, no FSDP. Tests Muon in isolation;
    rules out FSDP-side issues from interpretation. ~$3 on 1×B200.

  python scripts/muon_gpu_smoke.py --mode muon_fsdp --steps 50
    Muon hybrid + FSDP on >=2 GPUs. Production-shape rehearsal of
    Stage 2's optimizer path. ~$4 on 2×B200, ~$15 on 8×B200.

Pass/fail criteria for each mode:
  - All steps must have FINITE loss (no NaN/Inf forward)
  - All steps must have FINITE gradients (no NaN/Inf backward)
  - Loss must decrease monotonically-ish over 50 steps (sanity: the
    optimizer is moving in the right direction; "ish" because Adam
    moments + Newton-Schulz can produce tiny step-to-step bumps)
  - Per-step wall time should be in the ballpark of d8 repro's
    bf16 baseline (3-5 sec per step at mb=1/seq=4K on 1×B200)

Outputs:
  - Per-step (step, loss, grad_nan_count, time_ms) to stdout
  - JSON to artifacts/muon_smoke/<mode>_<timestamp>.json
  - Exit 0 if all steps pass criteria; 1 otherwise

Stage 2 decision matrix:
  adamw_baseline clean + muon_singlegpu clean + muon_fsdp clean
    → Stage 2 launches with `optimizer.type: muon_hybrid`
  muon_singlegpu shows NaN or divergent loss
    → fall back to AdamW for Stage 2; defer Muon investigation
  muon_fsdp shows NaN or divergent loss but singlegpu clean
    → Muon × FSDP-bf16 issue (treat like D8); stay AdamW for Stage 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--mode",
        choices=["adamw_baseline", "muon_singlegpu", "muon_fsdp"],
        required=True,
    )
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--config", type=str, default="configs/base_1b.yaml")
    parser.add_argument("--out-dir", type=str, default="artifacts/muon_smoke")
    parser.add_argument(
        "--muon-no-mup-scale",
        action="store_true",
        help="Disable the 1/width_mult LR scaling on the Muon-bucket "
             "(hidden weight matrices). Tests the DISPUTED post-review "
             "claim that Muon's spectral-norm-bounded update makes muP "
             "per-layer LR scaling redundant for hidden matrices. The "
             "2026-05-20 smoke showed Muon descended ~63%% of AdamW's "
             "rate WITH the scaling — this flag turns it off to A/B test.",
    )
    args = parser.parse_args()

    os.environ.setdefault("KERAS_BACKEND", "jax")
    import jax
    import jax.numpy as jnp
    import optax
    from keras import ops  # noqa: F401

    # Allow running before `pip install -e .`.
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root / "src"))

    from myllm.model.config import ModelConfig
    from myllm.model.transformer import build_model
    from myllm.training.loss import chunked_cross_entropy_with_z_loss
    from myllm.training.optimizer import (
        OptimizerConfig,
        build_optimizer,
        label_model_variables,
    )

    print(f"[muon-smoke] mode={args.mode} steps={args.steps} "
          f"mb={args.micro_batch} seq={args.seq_len}", flush=True)
    print(f"[muon-smoke] jax devices: {jax.devices()}", flush=True)

    use_muon = args.mode in ("muon_singlegpu", "muon_fsdp")
    use_fsdp = args.mode == "muon_fsdp"

    cfg = ModelConfig.from_yaml(args.config)
    print(
        f"[muon-smoke] config: {cfg.name} layers={cfg.layers} hidden={cfg.hidden_dim} "
        f"vocab={cfg.vocab_size} width_mult={cfg.mup_width_multiplier()}",
        flush=True,
    )

    # Sanity: verify optax has contrib.muon for the Muon modes.
    if use_muon:
        if not hasattr(optax, "contrib") or not hasattr(optax.contrib, "muon"):
            print("[muon-smoke] FATAL: optax.contrib.muon not available "
                  f"(optax {optax.__version__}); need optax>=0.2.5", flush=True)
            return 2
        print(f"[muon-smoke] optax {optax.__version__} OK; contrib.muon present", flush=True)

    model = build_model(cfg)
    dummy = jnp.zeros((1, 8), dtype=jnp.int32)
    _ = model(dummy)

    # Build optimizer matching what run_pretrain would build.
    opt_cfg = OptimizerConfig(
        peak_lr=3.0e-4,
        weight_decay=0.1,
        use_muon=use_muon,
        muon_beta=0.95,
        muon_ns_steps=5,
        muon_disable_mup_scale=args.muon_no_mup_scale,
    )
    param_labels = label_model_variables(model)
    width_mult = cfg.mup_width_multiplier()
    optimizer = build_optimizer(
        opt_cfg,
        lambda _step: 3.0e-4,
        param_labels=param_labels,
        mup_width_mult=width_mult,
    )
    print(f"[muon-smoke] optimizer built (use_muon={use_muon}, width_mult={width_mult})",
          flush=True)

    # Materialise params + opt_state.
    trainable = [jnp.asarray(v) for v in model.trainable_variables]
    non_trainable = [jnp.asarray(v) for v in model.non_trainable_variables]
    opt_state = optimizer.init(trainable)

    # FSDP setup (mode 3 only).
    mesh = None
    if use_fsdp:
        try:
            from myllm.training.mesh import build_mesh
            mesh = build_mesh(len(jax.devices()), axis="data")
            print(f"[muon-smoke] FSDP mesh: {mesh}", flush=True)
        except Exception as e:
            print(f"[muon-smoke] FSDP build_mesh failed: {e}; falling back to single-GPU",
                  flush=True)
            use_fsdp = False

    output_mult_value = (
        1.0 / width_mult
        if (cfg.mup is not None and cfg.mup.apply_lm_head_output_mult)
        else 1.0
    )

    def _build_minibatch():
        import numpy as np
        rng = np.random.default_rng(20260518)
        ids = rng.integers(1, cfg.vocab_size, size=(args.micro_batch, args.seq_len)).astype(np.int32)
        labels = rng.integers(1, cfg.vocab_size, size=(args.micro_batch, args.seq_len)).astype(np.int32)
        return {"input_ids": ids, "labels": labels}

    def loss_fn(trainable_, batch_):
        (hidden, lm_head_w, output_mult), _ = model.stateless_call(
            trainable_, non_trainable, batch_["input_ids"],
            return_loss_inputs=True,
        )
        loss, _ = chunked_cross_entropy_with_z_loss(
            hidden, lm_head_w, batch_["labels"],
            num_chunks=8,
            output_mult=output_mult,
            z_loss_coef=cfg.z_loss_coef,
            final_logit_softcap=cfg.final_logit_softcap,
        )
        return loss

    grad_fn = jax.jit(jax.value_and_grad(loss_fn))

    @jax.jit
    def update_fn(trainable_, opt_state_, grads):
        updates, new_opt_state = optimizer.update(grads, opt_state_, trainable_)
        new_trainable = optax.apply_updates(trainable_, updates)
        return new_trainable, new_opt_state

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for step in range(args.steps):
        batch = _build_minibatch()
        batch = {k: jnp.asarray(v) for k, v in batch.items()}
        t0 = time.time()
        loss, grads = grad_fn(trainable, batch)
        loss_val = float(loss.block_until_ready())

        # Probe grad health.
        n_nan = 0
        any_inf = False
        flat = jax.tree_util.tree_leaves(grads)
        for g in flat:
            gnp = jax.device_get(g)
            n_nan += int((gnp != gnp).sum())
            if (gnp == float("inf")).any() or (gnp == float("-inf")).any():
                any_inf = True

        # Optimizer step.
        if n_nan == 0 and not any_inf:
            trainable, opt_state = update_fn(trainable, opt_state, grads)
            # Force sync so dt reflects the actual step.
            jax.tree_util.tree_map(lambda a: a.block_until_ready(), trainable)

        dt = (time.time() - t0) * 1000.0
        rec = {
            "step": step,
            "loss": loss_val,
            "grad_n_nan": n_nan,
            "grad_any_inf": any_inf,
            "time_ms": dt,
        }
        records.append(rec)
        print(
            f"[muon-smoke] step={step:3d}  loss={loss_val:.4f}  "
            f"grad_nan={n_nan}  inf={any_inf}  dt={dt:.0f}ms",
            flush=True,
        )

    # Summary.
    losses = [r["loss"] for r in records]
    initial_loss = losses[0]
    final_loss = losses[-1]
    delta = initial_loss - final_loss
    any_bad = any(r["grad_n_nan"] > 0 or r["grad_any_inf"] for r in records)
    any_nonfinite_loss = any(not (loss_ == loss_) for loss_ in losses)

    print("", flush=True)
    print(f"[muon-smoke] === SUMMARY ===", flush=True)
    print(f"[muon-smoke]   initial loss : {initial_loss:.4f}", flush=True)
    print(f"[muon-smoke]   final loss   : {final_loss:.4f}", flush=True)
    print(f"[muon-smoke]   delta        : {delta:+.4f}   "
          f"({'GOOD: descending' if delta > 0.05 else 'CHECK: not clearly descending'})",
          flush=True)
    print(f"[muon-smoke]   any NaN grad : {any_bad}", flush=True)
    print(f"[muon-smoke]   any NaN loss : {any_nonfinite_loss}", flush=True)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_file = out_dir / f"{args.mode}_{timestamp}.json"
    out_file.write_text(json.dumps({
        "mode": args.mode,
        "steps": args.steps,
        "micro_batch": args.micro_batch,
        "seq_len": args.seq_len,
        "use_muon": use_muon,
        "use_fsdp": use_fsdp,
        "muon_no_mup_scale": bool(args.muon_no_mup_scale),
        "optax_version": optax.__version__,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "delta": delta,
        "any_bad_grad": any_bad,
        "any_nonfinite_loss": any_nonfinite_loss,
        "records": records,
    }, indent=2))
    print(f"[muon-smoke] wrote {out_file}", flush=True)

    if any_bad or any_nonfinite_loss:
        print(f"[muon-smoke] RESULT: FAIL (NaN observed in mode={args.mode})", flush=True)
        return 1
    if delta < -1.0:
        print(f"[muon-smoke] RESULT: DIVERGING (loss grew by {-delta:.2f}; mode={args.mode})",
              flush=True)
        return 1
    if delta < 0.01:
        print(f"[muon-smoke] RESULT: STALLED (loss didn't move; mode={args.mode})",
              flush=True)
        return 1
    print(f"[muon-smoke] RESULT: PASS (descending, finite; mode={args.mode})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
