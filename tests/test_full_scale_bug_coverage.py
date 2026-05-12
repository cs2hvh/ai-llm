"""Regression tests for the 4 gaps in 'full-scale-only bug' coverage.

The 2026-05-12 external reviewer enumerated 10 bug classes that only show
up at full scale (real corpus, multi-day runs, real checkpoints, real R2
uploads). Most are already covered by Phase A/B regression tests. The
four below were NOT yet covered:

  - #6  object-storage checkpoint partial write
  - #7  teacher-cache offset mismatch on resume
  - #9  shape mismatch at decay-phase activation
  - #10 quarantine path unavailable

Adding red tests here means that if any of these regress later (e.g.
someone removes the manifest-last atomicity, or restores data_position
without updating the position tracker, or makes the quarantine writer
crash on init), CI catches it before the next $20-30K base run.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# Pull in only what these tests need — heavy modules (Orbax, JAX) are
# imported lazily inside fixtures where required.


# --------------------------------------------------------------------------- #
# Bug #6: Object-storage checkpoint partial write
#
# Fault model:
#   step-NNNNNNN/state/ exists on disk (and on R2) because Orbax got
#   partway through serialization before the pod was preempted, but
#   manifest.json was never written. A naive resume picks the highest
#   step directory and tries to restore from a half-written state.
#
# Contract:
#   CheckpointManager.latest_complete_step() must skip step dirs that
#   exist but lack manifest.json. find_resume_step() must do the same.
#
# This is *the* atomicity contract — the entire R2 mirror story depends
# on the reader treating "no manifest" as "checkpoint doesn't exist".
# --------------------------------------------------------------------------- #
class TestCheckpointPartialWriteDetection:
    """The manifest-as-completion-marker contract.

    These tests don't need Orbax — they exercise the manifest-driven
    completion check in pure filesystem terms. That's the right scope:
    the bug pattern is about reader behaviour, not about Orbax
    serialization.
    """

    def _make_step_dir(self, root: Path, step: int, *, with_manifest: bool) -> Path:
        from myllm.training.checkpoint import CheckpointConfig, CheckpointManager
        cm = CheckpointManager(CheckpointConfig(root=str(root)))
        target = cm.step_dir(step)
        target.mkdir(parents=True, exist_ok=True)
        (target / "state").mkdir(exist_ok=True)
        # Drop a fake "state" file so the dir is plausibly non-empty,
        # the way Orbax would have left it mid-write.
        (target / "state" / "0.parquet").write_bytes(b"\x00" * 1024)
        if with_manifest:
            (target / "manifest.json").write_text(
                '{"step": ' + str(step) + ', "extra": {}}'
            )
        return target

    def test_latest_complete_skips_step_without_manifest(self, tmp_path):
        from myllm.training.checkpoint import CheckpointConfig, CheckpointManager

        self._make_step_dir(tmp_path, 100, with_manifest=True)
        # Step 200 was started but never finished: state/ exists, no manifest.
        self._make_step_dir(tmp_path, 200, with_manifest=False)

        cm = CheckpointManager(CheckpointConfig(root=str(tmp_path)))
        assert cm.latest_complete_step() == 100, (
            "must NOT promote half-written step 200 to 'latest complete' — "
            "this is the partial-write fault"
        )

    def test_list_complete_steps_excludes_partial(self, tmp_path):
        from myllm.training.checkpoint import CheckpointConfig, CheckpointManager

        self._make_step_dir(tmp_path, 100, with_manifest=True)
        self._make_step_dir(tmp_path, 200, with_manifest=False)
        self._make_step_dir(tmp_path, 300, with_manifest=True)

        cm = CheckpointManager(CheckpointConfig(root=str(tmp_path)))
        assert cm.list_complete_steps() == [100, 300]

    def test_find_resume_step_independent_helper_also_skips_partial(self, tmp_path):
        """find_resume_step is the lightweight helper used by run_pretrain
        before instantiating Orbax. It must enforce the same contract."""
        from myllm.training.checkpoint import find_resume_step

        self._make_step_dir(tmp_path, 500, with_manifest=True)
        self._make_step_dir(tmp_path, 600, with_manifest=False)

        assert find_resume_step(str(tmp_path)) == 500

    def test_manifest_written_last_in_save_order(self, tmp_path, monkeypatch):
        """The atomicity contract is: state/ first, manifest.json LAST.
        We verify the order by monkeypatching the Orbax save to fail mid-way
        and asserting that manifest.json doesn't exist on disk."""
        from myllm.training.checkpoint import (
            CheckpointConfig,
            CheckpointError,
            CheckpointManager,
        )

        cm = CheckpointManager(CheckpointConfig(root=str(tmp_path)))

        def _boom(*args, **kwargs):
            # Create the state dir to simulate Orbax writing some bytes
            # before exploding.
            target = cm.step_dir(42) / "state"
            target.mkdir(parents=True, exist_ok=True)
            (target / "0.parquet").write_bytes(b"\x00" * 32)
            raise RuntimeError("simulated orbax crash mid-save")

        monkeypatch.setattr(cm._orbax, "save", _boom)

        with pytest.raises(CheckpointError):
            cm.save(42, {"trainable_variables": {}})

        # Now: the state dir is on disk, but manifest must NOT be.
        assert cm.step_dir(42).exists()
        assert not (cm.step_dir(42) / "manifest.json").exists()
        # And the reader must treat it as a partial write.
        assert cm.latest_complete_step() is None


# --------------------------------------------------------------------------- #
# Bug #7: Teacher-cache offset mismatch on resume
#
# Fault model:
#   The pretrain run is at data_position = 1.5B tokens. The pod is
#   preempted; on resume, the loop loads state.data_position = 1.5B
#   but forgets to seed the SequentialCorpusPositions counter with
#   that value. The position tracker starts at 0; the teacher cache
#   reader fetches positions [0, B*S) but the batch is from corpus
#   position 1.5B+. Loss appears fine (it's still computed) but the
#   KL is wired to the wrong teacher rows. Silent corruption.
#
# Contract:
#   Whatever code resumes the loop MUST seed position_fn._pos from
#   state["data_position"], and MUST also persist data_position on
#   every batch.
# --------------------------------------------------------------------------- #
class TestTeacherCacheOffsetAlignment:
    """Position-tracker vs data_position alignment, both directions."""

    def test_position_tracker_resume_from_state(self):
        """When the loop hands a non-zero start_position to the tracker on
        resume, the tracker emits positions starting from there."""
        from myllm.training.decay_phase import SequentialCorpusPositions

        tracker = SequentialCorpusPositions(start_position=1_500_000_000)
        batch = {"input_ids": np.zeros((8, 1024), dtype=np.int32)}
        positions = tracker({}, batch)

        assert int(positions[0]) == 1_500_000_000, (
            "tracker started at 0 instead of resumed offset — this is the "
            "silent-corruption bug"
        )
        assert int(positions[-1]) == 1_500_000_000 + 8 * 1024 - 1
        assert tracker.position == 1_500_000_000 + 8 * 1024

    def test_zero_start_position_is_safe_default(self):
        """A fresh run (no resume) starts at 0."""
        from myllm.training.decay_phase import SequentialCorpusPositions
        tracker = SequentialCorpusPositions()
        batch = {"input_ids": np.zeros((4, 256), dtype=np.int32)}
        positions = tracker({}, batch)
        assert int(positions[0]) == 0
        assert tracker.position == 4 * 256

    def test_maybe_inject_uses_position_fn_output_verbatim(self):
        """maybe_inject must look up teacher rows at the exact positions
        returned by position_fn — no off-by-one, no rebasing."""
        from myllm.training.decay_phase import DecayPhaseActivation

        captured: dict = {}

        def position_fn(state, batch):
            n = batch["input_ids"].shape[0] * batch["input_ids"].shape[1]
            return np.arange(1_000_000, 1_000_000 + n, dtype=np.int64)

        reader = MagicMock()
        # Match shapes maybe_inject expects.
        reader.get_topk = MagicMock(
            side_effect=lambda pos: (
                np.zeros((1, len(pos), 8), dtype=np.float32),  # T=1, N, K=8
                np.zeros((1, len(pos), 8), dtype=np.uint32),
            )
        )

        def _spy(positions):
            captured["positions"] = positions
            return (
                np.zeros((1, len(positions), 8), dtype=np.float32),
                np.zeros((1, len(positions), 8), dtype=np.uint32),
            )
        reader.get_topk = _spy

        activation = DecayPhaseActivation(
            activation_step=0,  # always active
            reader=reader,
            position_fn=position_fn,
        )
        batch = {"input_ids": np.zeros((2, 4), dtype=np.int32)}
        out = activation.maybe_inject({"step": 100}, batch)

        # Bug catch: the reader was asked for the exact positions
        # position_fn produced, not for [0, N).
        assert captured["positions"][0] == 1_000_000
        assert captured["positions"][-1] == 1_000_000 + 2 * 4 - 1
        # And the teacher fields are injected with the right batch dims.
        assert out["teacher_topk_logits"].shape == (1, 2, 4, 8)

    def test_position_count_mismatch_raises_loudly(self):
        """If position_fn returns the wrong number of positions for the
        batch, maybe_inject must raise — not silently truncate (which
        would produce the cache-offset corruption all by itself)."""
        from myllm.training.decay_phase import DecayPhaseActivation

        def bad_position_fn(state, batch):
            # Returns 5 positions for a batch of B*S=8 — wrong.
            return np.arange(5, dtype=np.int64)

        reader = MagicMock()
        reader.get_topk = MagicMock(
            return_value=(
                np.zeros((1, 5, 8), dtype=np.float32),
                np.zeros((1, 5, 8), dtype=np.uint32),
            )
        )

        activation = DecayPhaseActivation(
            activation_step=0, reader=reader, position_fn=bad_position_fn,
        )
        batch = {"input_ids": np.zeros((2, 4), dtype=np.int32)}
        with pytest.raises(RuntimeError, match="out of sync"):
            activation.maybe_inject({"step": 1}, batch)


# --------------------------------------------------------------------------- #
# Bug #9: Shape mismatch at decay-phase activation
#
# Fault model:
#   train_step is jit-compiled on the stable-phase batch (no teacher
#   fields). At the activation step, the batch gains
#   ``teacher_topk_logits[T,B,S,K]`` + ``teacher_topk_indices[T,B,S,K]``.
#   If the batch dict's *added* keys aren't the expected ones, OR if
#   their shapes don't match the documented contract, downstream jit
#   recompile / shape errors will surface only at the activation step
#   (after 850k stable steps).
#
# Contract:
#   The only fields added at the activation boundary are exactly the
#   two teacher fields, with shapes (T, B, S, K) and dtypes float32 /
#   uint32 respectively. Any new field added by maybe_inject must come
#   with a schema bump (this test fails, forcing the change to be
#   conscious).
# --------------------------------------------------------------------------- #
class TestDecayActivationShapeStability:
    """Lock the decay-activation batch-schema contract."""

    def _activation(self, reader):
        from myllm.training.decay_phase import DecayPhaseActivation

        def position_fn(state, batch):
            n = batch["input_ids"].shape[0] * batch["input_ids"].shape[1]
            return np.arange(n, dtype=np.int64)

        return DecayPhaseActivation(
            activation_step=0, reader=reader, position_fn=position_fn,
        )

    def _reader_returning(self, T, K):
        m = MagicMock()
        m.get_topk = lambda pos: (
            np.zeros((T, len(pos), K), dtype=np.float32),
            np.zeros((T, len(pos), K), dtype=np.uint32),
        )
        return m

    def test_stable_phase_returns_input_batch_unchanged(self):
        """No new keys, no removed keys, identity preserved."""
        from myllm.training.decay_phase import DecayPhaseActivation
        activation = DecayPhaseActivation(
            activation_step=1000, reader=self._reader_returning(1, 8),
            position_fn=lambda s, b: np.arange(b["input_ids"].size),
        )
        batch = {
            "input_ids": np.zeros((2, 4), dtype=np.int32),
            "labels": np.zeros((2, 4), dtype=np.int32),
            "segment_ids": np.zeros((2, 4), dtype=np.int32),
            "loss_mask": np.ones((2, 4), dtype=np.float32),
        }
        out = activation.maybe_inject({"step": 500}, batch)
        # Identity: stable phase MUST return the *same* dict object.
        assert out is batch

    def test_decay_phase_adds_only_documented_fields(self):
        """Catches accidental schema growth (e.g. someone adding a 4th
        teacher field without updating train_step's compile signature)."""
        activation = self._activation(self._reader_returning(1, 8))
        batch = {
            "input_ids": np.zeros((2, 4), dtype=np.int32),
            "labels": np.zeros((2, 4), dtype=np.int32),
            "segment_ids": np.zeros((2, 4), dtype=np.int32),
            "loss_mask": np.ones((2, 4), dtype=np.float32),
        }
        out = activation.maybe_inject({"step": 100}, batch)

        added = set(out) - set(batch)
        assert added == {"teacher_topk_logits", "teacher_topk_indices"}, (
            f"unexpected schema change at activation: added {added}"
        )
        # Existing keys must survive unchanged.
        assert set(batch) <= set(out)

    def test_teacher_logits_shape_is_TBSK(self):
        activation = self._activation(self._reader_returning(2, 8))
        batch = {"input_ids": np.zeros((3, 16), dtype=np.int32)}
        out = activation.maybe_inject({"step": 100}, batch)
        # T=2 (set by reader), B=3, S=16, K=8
        assert out["teacher_topk_logits"].shape == (2, 3, 16, 8)
        assert out["teacher_topk_indices"].shape == (2, 3, 16, 8)

    def test_teacher_logits_dtype_is_float32(self):
        """The cache stores bf16-as-uint16 on disk; reader.get_topk()
        returns float32. The downstream KL loss assumes float32 — any
        regression to bf16 silently breaks softmax normalization."""
        activation = self._activation(self._reader_returning(1, 8))
        batch = {"input_ids": np.zeros((2, 4), dtype=np.int32)}
        out = activation.maybe_inject({"step": 100}, batch)
        assert out["teacher_topk_logits"].dtype == np.float32
        assert out["teacher_topk_indices"].dtype == np.uint32


# --------------------------------------------------------------------------- #
# Bug #10: Quarantine path unavailable
#
# Fault model:
#   Operator misconfigures the quarantine output path (or the volume
#   is read-only, or out of inodes). On the very first NaN spike,
#   QuarantineWriter blows up at __init__ time and kills the training
#   loop — losing not just the forensic dump but the entire run.
#
# Contract:
#   The quarantine writer is a forensic *aid*, not load-bearing. If
#   its path is unwritable, the loop must continue. Individual write()
#   calls already handle errors gracefully; the failure mode we need
#   to guard is __init__'s mkdir.
# --------------------------------------------------------------------------- #
class TestQuarantineGracefulDegradation:
    def test_write_does_not_raise_on_io_failure(self, tmp_path, monkeypatch):
        """An IOError mid-write must not propagate. Already covered by
        the broad try/except in write(); pinning the contract here."""
        from myllm.training.quarantine import QuarantineWriter

        q = QuarantineWriter(path=tmp_path / "q.jsonl")
        # Replace the JSONL file with something that won't accept writes:
        # we monkeypatch open() to raise for *this* path.
        real_open = open

        def boom(p, *a, **kw):
            if str(p) == str(q.path):
                raise OSError("disk full simulated")
            return real_open(p, *a, **kw)

        monkeypatch.setattr("builtins.open", boom)

        # Should NOT raise.
        q.write(
            step=10, data_position=0,
            batch={"input_ids": np.zeros((1, 4), dtype=np.int32)},
            loss=float("nan"),
        )
        # Incident counter should NOT increment when the write failed
        # (we should not pretend we recorded it).
        assert q.incident_count == 0

    def test_unwritable_parent_directory_does_not_crash_init(
        self, tmp_path, monkeypatch
    ):
        """If the parent directory can't be created (read-only volume,
        permission denied, out of inodes), construction must NOT propagate
        the OSError — the loop must keep running, just without quarantine.

        We monkeypatch Path.mkdir to raise (chmod can't be used here:
        the test process often runs as root in CI / dev containers,
        which bypasses chmod restrictions).
        """
        from myllm.training import quarantine as q_mod

        original_mkdir = Path.mkdir

        def boom(self, *a, **kw):
            # Only fail for the quarantine-target parent — let other
            # tmp_path mkdir calls (pytest internals) succeed.
            if "q.jsonl" in str(self) or self.name == "ro":
                raise PermissionError("simulated read-only volume")
            return original_mkdir(self, *a, **kw)

        monkeypatch.setattr(Path, "mkdir", boom)

        # Construction must not raise.
        q = q_mod.QuarantineWriter(path=tmp_path / "ro" / "q.jsonl")
        # Subsequent writes must also be safe (already covered by
        # write's broad except, but worth pinning here for the
        # full degraded-mode round-trip).
        q.write(
            step=1, data_position=0,
            batch={"input_ids": np.zeros((1, 4), dtype=np.int32)},
            loss=1.0,
        )
        # Nothing should have been recorded.
        assert q.incident_count == 0
