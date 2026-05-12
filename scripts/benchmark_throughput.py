#!/usr/bin/env python3
"""Throughput + MFU benchmark for the MyLLM training stack.

Per 2026-05-12 reviewer Q&A §6 + plan_v3 §2.7: don't trust theoretical
projections; measure real tok/sec/GPU on the actual hardware you'll
use. This script wraps the existing training loop with measurement
hooks and emits a structured JSON report.

What it measures:
  - Steady-state tok/sec/GPU (after a warmup window so JIT compile,
    optimizer warmup, and dataloader spin-up don't pollute the number)
  - Aggregate tok/sec across all visible JAX devices
  - Peak GPU memory (HBM)
  - Time per step (mean + p50 + p95 + p99)
  - **MFU** — Model FLOPs Utilization, computed from
    forward_flops_per_token × tokens_per_sec / peak_device_flops_bf16

Usage:

    # Synthetic data (no corpus needed — fastest validation):
    KERAS_BACKEND=jax python scripts/benchmark_throughput.py \\
        --model-config configs/pilot_250m.yaml \\
        --tokenizer-path artifacts/tokenizer_v1.json \\
        --synthetic-data \\
        --warmup-steps 50 \\
        --measure-steps 200 \\
        --output benchmark_pilot_synthetic.json

    # Real packed corpus (catches dataloader overhead):
    KERAS_BACKEND=jax python scripts/benchmark_throughput.py \\
        --model-config configs/base_1b.yaml \\
        --tokenizer-path artifacts/tokenizer_v1.json \\
        --packed-corpus-root /workspace/corpus_pilot/train \\
        --warmup-steps 100 \\
        --measure-steps 500 \\
        --output benchmark_1b_real.json

Per the reviewer: a 2-hr benchmark on real packed corpus is what
anchors the cost model. Use --measure-steps to control duration.

Output JSON shape:

    {
      "config": {...},
      "device": {
        "name": "NVIDIA H200",
        "peak_flops_bf16": 1979e12,
        "n_devices": 1
      },
      "measurements": {
        "warmup_steps": 50,
        "measure_steps": 200,
        "mean_step_time_sec": 0.34,
        "p50_step_time_sec": 0.33,
        "p95_step_time_sec": 0.36,
        "p99_step_time_sec": 0.41,
        "tokens_per_sec_per_device": 24117,
        "tokens_per_sec_aggregate": 24117,
        "peak_memory_gb": 19.4
      },
      "mfu": {
        "estimate_pct": 38.7,
        "model_flops_per_token": 1.65e9,
        "compute": "6 * num_params (Chinchilla approximation)"
      }
    }
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

# Keras backend before any keras import.
os.environ.setdefault("KERAS_BACKEND", "jax")


# Peak bf16 tensor-core FLOPS per device, for MFU computation.
# Numbers from the device's spec sheet (NOT manufacturer max-theoretical).
# Sources cited in plan_v3 §2.8.
_PEAK_FLOPS_BF16 = {
    "H100": 989e12,        # H100 SXM, bf16 with sparsity
    "H200": 1979e12,       # H200 SXM, bf16 dense
    "B200": 5000e12,       # B200 dense bf16 estimate
    "A100": 312e12,        # A100, bf16
    "L4": 30.3e12,         # L4 inference card, bf16
    "L40S": 91.6e12,       # L40S
    "RTX_4090": 165e12,    # Consumer 4090, bf16
}


def _detect_device_name() -> str:
    """Try to identify the GPU via nvidia-smi → maps to our PEAK_FLOPS keys.

    Returns 'UNKNOWN' on detection failure; caller can then pass
    --peak-flops-bf16 explicitly to compute MFU.
    """
    import subprocess
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
        ).decode().strip().splitlines()
        if not out:
            return "UNKNOWN"
        name = out[0]
        # Normalize: "NVIDIA H200" → "H200", "NVIDIA H100 80GB HBM3" → "H100"
        name_up = name.upper()
        for key in _PEAK_FLOPS_BF16:
            if key.upper() in name_up:
                return key
        return name
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "UNKNOWN"


def _peak_gpu_memory_bytes() -> int:
    """Total bytes used on device 0 — best-effort across JAX versions."""
    try:
        import jax
        dev = jax.devices()[0]
        if hasattr(dev, "memory_stats"):
            stats = dev.memory_stats()
            return int(stats.get("peak_bytes_in_use", 0))
    except Exception:  # noqa: BLE001
        pass
    return 0


def _model_flops_per_token(n_params: int) -> float:
    """Chinchilla forward-pass approximation: 2N FLOPs/token (forward),
    6N FLOPs/token (forward + backward + weight update).

    Args:
        n_params: total trainable parameter count.

    Returns:
        FLOPs per token for ONE training step (forward + backward + update).
    """
    return 6.0 * n_params


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-config", required=True)
    p.add_argument("--tokenizer-path", required=True)

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--synthetic-data", action="store_true",
                     help="Use random tokens (fastest; isolates compute).")
    src.add_argument("--packed-corpus-root",
                     help="Use a packed corpus (catches dataloader overhead).")

    p.add_argument("--warmup-steps", type=int, default=50,
                   help="Steps to skip before measurement begins (lets JIT "
                        "compile + optimizer warmup settle). Default 50.")
    p.add_argument("--measure-steps", type=int, default=200,
                   help="Steps to measure throughput over. Default 200.")
    p.add_argument("--micro-batch", type=int, default=None,
                   help="Override micro_batch_per_device. Default: read from configs.")
    p.add_argument("--peak-lr", type=float, default=2e-4,
                   help="Peak LR. Irrelevant for throughput but required by loop.")
    p.add_argument("--peak-flops-bf16", type=float, default=None,
                   help="Override device peak bf16 FLOPS for MFU calc.")
    p.add_argument("--output", default=None,
                   help="Write the JSON report to this path. Default: stdout.")
    args = p.parse_args()

    if not args.synthetic_data and not args.packed_corpus_root:
        print("ERROR: --synthetic-data or --packed-corpus-root required",
              file=sys.stderr)
        return 2

    # Import after KERAS_BACKEND is set.
    import numpy as np
    import jax
    from myllm.model.config import ModelConfig
    from myllm.utils import configure_logging, get_logger
    configure_logging()
    log = get_logger(__name__)

    model_cfg = ModelConfig.from_yaml(args.model_config)

    # Build the data iterator. Reuse the same patterns as run_pretrain.py
    # but in-process so we can time each step precisely.
    if args.synthetic_data:
        from myllm.data.synthetic import make_synthetic_data_iter
        micro_batch = args.micro_batch or 8
        seq_len = model_cfg.context_length
        data_iter = make_synthetic_data_iter(
            micro_batch=micro_batch,
            sequence_length=seq_len,
            vocab_size=model_cfg.vocab_size,
            n_steps=args.warmup_steps + args.measure_steps + 10,
            seed=42,
        )
    else:
        from myllm.data.packed_corpus import PackedCorpusReader, iter_packed_pairs

        reader = PackedCorpusReader(args.packed_corpus_root)
        micro_batch = args.micro_batch or 8
        seq_len = reader.sequence_length - 1  # input is seq_len - 1 after shift
        pair_iter = iter_packed_pairs(reader, start_sequence_id=0)
        # Wrap into the batch dict format the loop expects.
        from scripts.run_pretrain import batch_pairs
        data_iter = batch_pairs(pair_iter, micro_batch, seq_len)

    # Build model + optimizer.
    from myllm.training.optimizer import OptimizerConfig
    from scripts.run_pretrain import init_model_and_optimizer, initial_train_state
    opt_cfg = OptimizerConfig(peak_lr=args.peak_lr)
    model, optimizer = init_model_and_optimizer(
        model_cfg, opt_cfg, total_steps=args.warmup_steps + args.measure_steps,
        lr_schedule_cfg=None,
    )
    state = initial_train_state(model, optimizer)
    n_params = int(sum(int(np.asarray(v).size) for v in model.weights))

    # Train step.
    from myllm.training.train_step import make_train_step
    train_step = make_train_step(model, optimizer)

    log.info(
        "benchmark_start",
        n_params=n_params,
        micro_batch=micro_batch,
        seq_len=seq_len,
        warmup=args.warmup_steps,
        measure=args.measure_steps,
        device=str(jax.devices()),
    )

    # Run loop with per-step timing.
    step_times: list[float] = []
    losses: list[float] = []
    n_devices = len(jax.devices())
    tokens_per_step = micro_batch * seq_len * n_devices

    total_steps = args.warmup_steps + args.measure_steps
    iterator = iter(data_iter)
    for step in range(total_steps):
        try:
            batch = next(iterator)
        except StopIteration:
            log.warning("benchmark_data_exhausted_early", step=step)
            break
        t0 = time.perf_counter()
        state, metrics = train_step(state, batch)
        # Sync to ensure JAX async dispatch completes before timing.
        jax.block_until_ready(state.get("trainable_variables") or state)
        dt = time.perf_counter() - t0
        if step >= args.warmup_steps:
            step_times.append(dt)
            losses.append(float(metrics.get("loss", 0)))
        if step % 25 == 0:
            log.info(
                "benchmark_step",
                step=step,
                step_time_sec=round(dt, 4),
                loss=float(metrics.get("loss", 0)),
                in_warmup=step < args.warmup_steps,
            )

    if not step_times:
        print("ERROR: no measure-window steps completed", file=sys.stderr)
        return 1

    # Stats.
    step_times.sort()
    mean_t = sum(step_times) / len(step_times)
    p50 = step_times[len(step_times) // 2]
    p95 = step_times[int(len(step_times) * 0.95)]
    p99 = step_times[int(len(step_times) * 0.99)]
    tps_per_device = tokens_per_step / mean_t / max(1, n_devices)
    tps_aggregate = tokens_per_step / mean_t
    peak_mem = _peak_gpu_memory_bytes()

    # MFU.
    device_name = _detect_device_name()
    peak_flops = args.peak_flops_bf16 or _PEAK_FLOPS_BF16.get(device_name, 0)
    flops_per_token = _model_flops_per_token(n_params)
    mfu_pct = 0.0
    if peak_flops > 0:
        mfu_pct = (flops_per_token * tps_per_device) / peak_flops * 100

    report = {
        "config": {
            "model_config": args.model_config,
            "synthetic_data": args.synthetic_data,
            "packed_corpus_root": args.packed_corpus_root,
            "micro_batch_per_device": micro_batch,
            "sequence_length": seq_len,
            "tokens_per_step": tokens_per_step,
            "warmup_steps": args.warmup_steps,
            "measure_steps": len(step_times),
            "n_params": n_params,
        },
        "device": {
            "name": device_name,
            "n_devices": n_devices,
            "peak_flops_bf16": peak_flops,
            "jax_devices": [str(d) for d in jax.devices()],
        },
        "measurements": {
            "mean_step_time_sec": round(mean_t, 4),
            "p50_step_time_sec": round(p50, 4),
            "p95_step_time_sec": round(p95, 4),
            "p99_step_time_sec": round(p99, 4),
            "tokens_per_sec_per_device": round(tps_per_device, 0),
            "tokens_per_sec_aggregate": round(tps_aggregate, 0),
            "peak_memory_bytes": peak_mem,
            "peak_memory_gb": round(peak_mem / 1e9, 2) if peak_mem else None,
            "mean_loss": round(sum(losses) / len(losses), 4) if losses else None,
        },
        "mfu": {
            "estimate_pct": round(mfu_pct, 1),
            "model_flops_per_token": flops_per_token,
            "compute_method": "6 * n_params (Chinchilla approximation: fwd + bwd + update)",
        },
        "cost_model_anchor": {
            "tok_per_sec_per_gpu": round(tps_per_device, 0),
            "extrapolation_1T_tokens_one_device_hours": round(
                1_000_000_000_000 / tps_per_device / 3600, 1
            ) if tps_per_device > 0 else None,
            "extrapolation_1T_tokens_8gpu_hours": round(
                1_000_000_000_000 / (tps_per_device * 8) / 3600, 1
            ) if tps_per_device > 0 else None,
        },
    }

    out_text = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(out_text)
        print(f"report written to {args.output}")
        # Also print headline numbers to stdout for human-readable scan.
        m = report["measurements"]
        d = report["device"]
        mfu = report["mfu"]
        c = report["cost_model_anchor"]
        print()
        print(f"=== headline ===")
        print(f"  device:            {d['name']} × {d['n_devices']}")
        print(f"  tok/sec/device:    {m['tokens_per_sec_per_device']:,.0f}")
        print(f"  tok/sec/aggregate: {m['tokens_per_sec_aggregate']:,.0f}")
        print(f"  MFU estimate:      {mfu['estimate_pct']:.1f}%")
        print(f"  peak HBM:          {m['peak_memory_gb']} GB")
        if c["extrapolation_1T_tokens_8gpu_hours"]:
            print(f"  1T @ 8×{d['name']}:  ~{c['extrapolation_1T_tokens_8gpu_hours']:.0f} hours")
    else:
        print(out_text)

    return 0


if __name__ == "__main__":
    sys.exit(main())
