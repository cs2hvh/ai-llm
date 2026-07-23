"""Regression tests for the 2026-07-23 throughput-accounting audit fix.

The bug: benchmark_throughput.py multiplied global-batch tokens by device
count after the batch had already been sharded — inflating aggregate
throughput and MFU by ~world size. These tests pin the corrected invariants
across world sizes 1/2/4/8. Pure python — no JAX required.
"""
import sys
from pathlib import Path

import pytest

# Standalone-run fallback (suite normally runs with the package installed).
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from myllm.utils.accounting import ThroughputReport, compute_throughput, mfu_pct  # noqa: E402


@pytest.mark.parametrize("n_devices", [1, 2, 4, 8])
def test_global_tokens_counted_exactly_once(n_devices):
    """Aggregate depends ONLY on the global batch, never on world size."""
    r = compute_throughput(
        global_batch_sequences=16, sequence_length=8192,
        n_devices=n_devices, mean_step_time_sec=2.0,
    )
    assert r.tokens_per_step == 16 * 8192  # no n_devices factor, ever
    assert r.tokens_per_sec_aggregate == pytest.approx(16 * 8192 / 2.0)
    # per-device is derived by DIVISION:
    assert r.tokens_per_sec_per_device == pytest.approx(r.tokens_per_sec_aggregate / n_devices)
    # invariant: per_device * n == aggregate exactly
    assert r.tokens_per_sec_per_device * n_devices == pytest.approx(r.tokens_per_sec_aggregate)


def test_docstring_example_from_execution_plan():
    """The execution plan's worked example (§P1-41): global batch 4 × 8192 @ 1s
    → aggregate 32,768 tok/s; per-device on 4 GPUs = 8,192 tok/s."""
    r = compute_throughput(4, 8192, 4, 1.0)
    assert r.tokens_per_step == 32_768
    assert r.tokens_per_sec_aggregate == pytest.approx(32_768.0)
    assert r.tokens_per_sec_per_device == pytest.approx(8_192.0)


def test_legacy_bug_magnitude_c2_case():
    """The C2 case: mb=16 (global), seq=8192, 4×B200, step 2.37s.
    Buggy formula reported ×4 the true aggregate — i.e. the published
    221,246 tok/s was really ~55.3K tok/s (and 30% MFU really ~7.5%)."""
    r = compute_throughput(16, 8192, 4, 2.37)
    buggy_aggregate = 16 * 8192 * 4 / 2.37
    assert buggy_aggregate == pytest.approx(221_218, rel=1e-3)      # what was reported
    assert r.tokens_per_sec_aggregate == pytest.approx(55_304, rel=1e-3)  # the truth
    assert buggy_aggregate / r.tokens_per_sec_aggregate == pytest.approx(4.0)


def test_mfu_uses_true_per_device_rate():
    r = compute_throughput(16, 8192, 4, 2.37)
    # 1B model, 6N FLOPs/token, B200 bf16 peak 2.25e15
    mfu = mfu_pct(r.tokens_per_sec_per_device, 6 * 1.24e9, 2.25e15)
    assert mfu == pytest.approx(4.57, abs=0.1)  # NOT ~18% (buggy ×4) nor 30% (published)


def test_single_device_degenerate_case():
    r = compute_throughput(8, 1024, 1, 0.5)
    assert r.tokens_per_sec_per_device == r.tokens_per_sec_aggregate


@pytest.mark.parametrize(
    "kwargs",
    [
        {"global_batch_sequences": 0, "sequence_length": 1, "n_devices": 1, "mean_step_time_sec": 1.0},
        {"global_batch_sequences": 1, "sequence_length": 0, "n_devices": 1, "mean_step_time_sec": 1.0},
        {"global_batch_sequences": 1, "sequence_length": 1, "n_devices": 0, "mean_step_time_sec": 1.0},
        {"global_batch_sequences": 1, "sequence_length": 1, "n_devices": 1, "mean_step_time_sec": 0.0},
    ],
)
def test_invalid_inputs_rejected(kwargs):
    with pytest.raises(ValueError):
        compute_throughput(**kwargs)


def test_report_is_frozen():
    r = compute_throughput(1, 1, 1, 1.0)
    assert isinstance(r, ThroughputReport)
    with pytest.raises(Exception):
        r.tokens_per_step = 999  # frozen dataclass
