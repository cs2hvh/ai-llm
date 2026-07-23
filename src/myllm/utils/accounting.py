"""Throughput/token accounting — pure math, framework-free.

HISTORY (2026-07-23 audit): the original benchmark_throughput.py computed
``tokens_per_step = micro_batch * seq_len * n_devices``. But ``batch_pairs``
builds ONE global ``[micro_batch, seq]`` batch which the jitted train_step
receives under ``data_sharding`` — i.e. it is SPLIT across the mesh data
axis, not replicated. Multiplying by ``n_devices`` therefore double-counted
by exactly world size: reported aggregate throughput and MFU were inflated
~n_devices (legacy C2 "30%/46% MFU on 4xB200" was really ~7.5%/11.5%).

Rule (program invariant): **global tokens are counted exactly once.**
``micro_batch`` here is always the GLOBAL number of sequences in one
optimizer step.  Per-device numbers are derived by division, never by
multiplication of a global quantity.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ThroughputReport:
    global_batch_sequences: int
    sequence_length: int
    n_devices: int
    mean_step_time_sec: float
    # Derived — the only legal derivations:
    tokens_per_step: int              # global_batch_sequences * sequence_length
    tokens_per_sec_aggregate: float   # tokens_per_step / mean_step_time
    tokens_per_sec_per_device: float  # aggregate / n_devices


def compute_throughput(
    global_batch_sequences: int,
    sequence_length: int,
    n_devices: int,
    mean_step_time_sec: float,
) -> ThroughputReport:
    """Compute throughput from a GLOBAL batch. Never multiply by world size.

    Invariants (regression-tested):
      tokens_per_step        == global_batch_sequences * sequence_length
      aggregate              == tokens_per_step / mean_step_time
      per_device * n_devices == aggregate  (exactly)
    """
    if global_batch_sequences <= 0 or sequence_length <= 0:
        raise ValueError("batch and sequence length must be positive")
    if n_devices <= 0:
        raise ValueError("n_devices must be positive")
    if mean_step_time_sec <= 0:
        raise ValueError("mean_step_time_sec must be positive")

    tokens_per_step = global_batch_sequences * sequence_length
    aggregate = tokens_per_step / mean_step_time_sec
    per_device = aggregate / n_devices
    return ThroughputReport(
        global_batch_sequences=global_batch_sequences,
        sequence_length=sequence_length,
        n_devices=n_devices,
        mean_step_time_sec=mean_step_time_sec,
        tokens_per_step=tokens_per_step,
        tokens_per_sec_aggregate=aggregate,
        tokens_per_sec_per_device=per_device,
    )


def mfu_pct(
    tokens_per_sec_per_device: float,
    flops_per_token: float,
    peak_flops_per_device: float,
) -> float:
    """MFU is SECONDARY to measured throughput and uses true per-device tok/s.

    ``flops_per_token`` must come from a versioned FLOP model; the 6N
    approximation is not valid as the sole model at long context.
    """
    if peak_flops_per_device <= 0:
        return 0.0
    return (flops_per_token * tokens_per_sec_per_device) / peak_flops_per_device * 100.0
