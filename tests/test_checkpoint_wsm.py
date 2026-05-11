"""Regression tests for WSM (Warmup-Stable-Merge) checkpoint averaging.

R5 from the 2026-05-11 research dossier. Averaging the last N stable-phase
checkpoints produces a model that outperforms the WSD-decayed final per
arXiv:2507.17634.

These tests don't validate the *quality* claim (that needs real training).
They validate the *mechanical correctness* of the merge: weights average
elementwise; non-weight state inherits from the latest source.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

# Skip the entire module if jax isn't available.
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
np = pytest.importorskip("numpy")

from myllm.training.checkpoint import CheckpointConfig, CheckpointManager


@pytest.fixture
def cm():
    """A fresh CheckpointManager in a temp dir, with retention disabled."""
    with tempfile.TemporaryDirectory() as tmp:
        config = CheckpointConfig(root=tmp, keep_last_n=100, keep_every_n=1)
        yield CheckpointManager(config)


def _state(weights: np.ndarray, step: int) -> dict:
    """Build a minimal state PyTree with one weight tensor."""
    return {
        "trainable_variables": [jnp.asarray(weights)],
        "non_trainable_variables": [],
        "step": step,
        "opt_state": {"momentum": jnp.asarray(np.full_like(weights, float(step)))},
    }


def test_wsm_merge_two_checkpoints_averages_weights(cm):
    """Merging [1,2,3] and [3,4,5] should produce [2,3,4]."""
    cm.save(100, _state(np.array([1.0, 2.0, 3.0]), step=100))
    cm.save(200, _state(np.array([3.0, 4.0, 5.0]), step=200))
    cm.merge_checkpoints([100, 200], output_step=999)

    merged = cm.restore(999)
    w = np.asarray(merged["trainable_variables"][0])
    np.testing.assert_array_almost_equal(w, np.array([2.0, 3.0, 4.0]))


def test_wsm_merge_three_checkpoints_uniform_mean(cm):
    """Three checkpoints with distinct values average element-wise."""
    cm.save(10, _state(np.array([1.0, 2.0]), step=10))
    cm.save(20, _state(np.array([5.0, 6.0]), step=20))
    cm.save(30, _state(np.array([9.0, 10.0]), step=30))
    cm.merge_checkpoints([10, 20, 30], output_step=999)

    w = np.asarray(cm.restore(999)["trainable_variables"][0])
    np.testing.assert_array_almost_equal(w, np.array([5.0, 6.0]))


def test_wsm_merge_inherits_opt_state_from_latest(cm):
    """Optimizer state should come from the LAST source, not averaged."""
    cm.save(100, _state(np.array([1.0]), step=100))   # opt_state.momentum = [100.0]
    cm.save(200, _state(np.array([1.0]), step=200))   # opt_state.momentum = [200.0]
    cm.merge_checkpoints([100, 200], output_step=999)

    merged = cm.restore(999)
    mom = np.asarray(merged["opt_state"]["momentum"])
    # Must match step=200's opt_state, NOT the average (150).
    np.testing.assert_array_almost_equal(mom, np.array([200.0]))


def test_wsm_merge_records_provenance_in_manifest(cm):
    """The merged checkpoint's manifest should record source steps."""
    cm.save(10, _state(np.array([1.0]), step=10))
    cm.save(20, _state(np.array([2.0]), step=20))
    target = cm.merge_checkpoints([10, 20], output_step=999)

    from myllm.utils.io import read_json
    manifest = read_json(target / "manifest.json")
    assert manifest["step"] == 999
    assert manifest["extra"]["wsm_merged"] is True
    assert manifest["extra"]["source_steps"] == [10, 20]
    assert manifest["extra"]["source_count"] == 2


def test_wsm_merge_recent_picks_last_n_plain_checkpoints(cm):
    """merge_recent(2) should pick the 2 most recent NON-merged checkpoints."""
    for s in (10, 20, 30, 40):
        cm.save(s, _state(np.array([float(s)]), step=s))
    cm.merge_recent(n=2, output_step=999)

    merged = cm.restore(999)
    w = np.asarray(merged["trainable_variables"][0])
    # Should average step 30 and step 40.
    np.testing.assert_array_almost_equal(w, np.array([35.0]))


def test_wsm_merge_excludes_prior_merged_from_source_list(cm):
    """merge_recent must not re-merge an already-merged checkpoint."""
    for s in (10, 20, 30):
        cm.save(s, _state(np.array([float(s)]), step=s))
    # First merge: 20 + 30 → 999 (average 25).
    cm.merge_recent(n=2, output_step=999)
    # Add a fresh plain checkpoint and merge again.
    cm.save(40, _state(np.array([40.0]), step=40))
    # merge_recent(n=2) must pick steps 30 and 40 (not 30 and 999, even though
    # 999 has a higher step number).
    cm.merge_recent(n=2, output_step=1000)

    w = np.asarray(cm.restore(1000)["trainable_variables"][0])
    np.testing.assert_array_almost_equal(w, np.array([35.0]))


def test_wsm_merge_rejects_single_source(cm):
    """A merge of one checkpoint is undefined; must raise."""
    cm.save(10, _state(np.array([1.0]), step=10))
    with pytest.raises(ValueError, match="(?i)needs.*>=.*2"):
        cm.merge_checkpoints([10], output_step=999)
