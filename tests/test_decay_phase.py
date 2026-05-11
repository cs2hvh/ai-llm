"""Regression tests for decay-phase distillation activation (R0 final piece).

Validates the loop-side activation switch:
  - `SequentialCorpusPositions`: counts up correctly, persists internally.
  - `DecayPhaseActivation.is_active`: gates on (step, reader) jointly.
  - `maybe_inject`: pass-through in stable phase, augments batch in decay.
  - Augmented batch shape: T x B x S x K teacher tensors.
  - YAML loader: activation_step = round(activation_fraction * total_steps).
  - Failure modes: missing position_fn while active → loud RuntimeError;
    position/batch shape mismatch → loud RuntimeError.

Uses synthetic cache (via the existing teacher_cache writer) so no real
teacher model needed.
"""
from __future__ import annotations

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pa = pytest.importorskip("pyarrow")

from myllm.data.teacher_cache import (
    CacheManifest,
    CacheShard,
    MultiTeacherCacheReader,
    ShardManifestEntry,
    TeacherCacheReader,
    compute_shard_key,
    write_manifest,
    write_shard,
)
from myllm.training.decay_phase import (
    DecayPhaseActivation,
    SequentialCorpusPositions,
)


def _build_reader(tmp_path: Path, teacher_id: str, n_tokens: int, top_k: int):
    """Build a single-shard synthetic cache + return a TeacherCacheReader."""
    output_dir = tmp_path / teacher_id
    rng = np.random.default_rng(seed=hash(teacher_id) & 0xFFFFFFFF)
    logits = rng.integers(0, 65535, size=(n_tokens, top_k), dtype="uint16")
    indices = rng.integers(0, 131072, size=(n_tokens, top_k), dtype="uint32")
    shard = CacheShard(
        teacher_id=teacher_id, corpus_sha256="0" * 64, tokenizer_sha256="0" * 64,
        start_token_position=0, end_token_position=n_tokens, top_k=top_k,
        logits=logits, indices=indices,
    )
    key = compute_shard_key(teacher_id, top_k, "0" * 64, 0, n_tokens)
    sha = write_shard(shard, output_dir / key)
    manifest = CacheManifest(
        teacher_id=teacher_id, corpus_sha256="0" * 64, tokenizer_sha256="0" * 64,
        top_k=top_k,
        shards=[ShardManifestEntry(0, n_tokens, key, sha)],
    )
    manifest_path = output_dir / f"{teacher_id}_manifest.json"
    write_manifest(manifest, manifest_path)
    return TeacherCacheReader(teacher_id, manifest_path, output_dir)


# --------------------------------------------------------------------------- #
# SequentialCorpusPositions
# --------------------------------------------------------------------------- #
class TestSequentialPositions:
    def test_starts_at_zero(self):
        tracker = SequentialCorpusPositions()
        assert tracker.position == 0

    def test_advances_by_batch_size_x_seq_len(self):
        tracker = SequentialCorpusPositions()
        batch = {"input_ids": np.zeros((2, 8), dtype=np.int32)}
        positions = tracker(state={}, batch=batch)
        assert positions.tolist() == list(range(16))
        assert tracker.position == 16

        # Second call continues where the first stopped.
        positions2 = tracker(state={}, batch=batch)
        assert positions2.tolist() == list(range(16, 32))
        assert tracker.position == 32

    def test_resume_from_persisted_offset(self):
        tracker = SequentialCorpusPositions(start_position=1_000_000)
        batch = {"input_ids": np.zeros((2, 4), dtype=np.int32)}
        positions = tracker(state={}, batch=batch)
        assert positions.tolist() == list(range(1_000_000, 1_000_008))


# --------------------------------------------------------------------------- #
# DecayPhaseActivation.is_active
# --------------------------------------------------------------------------- #
class TestIsActive:
    def _reader(self, tmp_path):
        r_a = _build_reader(tmp_path / "ta", "ta", n_tokens=32, top_k=4)
        return MultiTeacherCacheReader([r_a])

    def test_inactive_when_reader_none(self, tmp_path):
        activation = DecayPhaseActivation(
            activation_step=100, reader=None, position_fn=None
        )
        assert activation.is_active({"step": 200}) is False

    def test_inactive_before_activation_step(self, tmp_path):
        activation = DecayPhaseActivation(
            activation_step=100, reader=self._reader(tmp_path), position_fn=None
        )
        assert activation.is_active({"step": 50}) is False
        assert activation.is_active({"step": 99}) is False

    def test_active_at_activation_step(self, tmp_path):
        activation = DecayPhaseActivation(
            activation_step=100, reader=self._reader(tmp_path), position_fn=None
        )
        assert activation.is_active({"step": 100}) is True
        assert activation.is_active({"step": 100_000}) is True


# --------------------------------------------------------------------------- #
# maybe_inject
# --------------------------------------------------------------------------- #
class TestMaybeInject:
    def _build(self, tmp_path, n_teachers=2, top_k=4, n_corpus=64):
        readers = [
            _build_reader(tmp_path / f"t{i}", f"t{i}", n_tokens=n_corpus, top_k=top_k)
            for i in range(n_teachers)
        ]
        multi = MultiTeacherCacheReader(readers)
        return multi

    def test_pass_through_in_stable_phase(self, tmp_path):
        multi = self._build(tmp_path)
        activation = DecayPhaseActivation(
            activation_step=1000,
            reader=multi,
            position_fn=SequentialCorpusPositions(),
        )
        batch = {
            "input_ids": np.zeros((2, 4), dtype=np.int32),
            "labels": np.zeros((2, 4), dtype=np.int32),
        }
        out = activation.maybe_inject({"step": 0}, batch)
        # Stable phase: batch is unchanged (same identity, no teacher fields).
        assert out is batch
        assert "teacher_topk_logits" not in out

    def test_injects_teacher_fields_in_decay_phase(self, tmp_path):
        multi = self._build(tmp_path, n_teachers=2, top_k=4, n_corpus=64)
        activation = DecayPhaseActivation(
            activation_step=10,
            reader=multi,
            position_fn=SequentialCorpusPositions(),
        )
        batch = {
            "input_ids": np.zeros((2, 4), dtype=np.int32),  # B=2, S=4 → 8 positions
            "labels": np.zeros((2, 4), dtype=np.int32),
        }
        out = activation.maybe_inject({"step": 10}, batch)
        assert "teacher_topk_logits" in out
        assert "teacher_topk_indices" in out
        # Shape: [T, B, S, K] = [2, 2, 4, 4]
        assert out["teacher_topk_logits"].shape == (2, 2, 4, 4)
        assert out["teacher_topk_indices"].shape == (2, 2, 4, 4)

    def test_disabled_when_reader_is_none(self, tmp_path):
        activation = DecayPhaseActivation(
            activation_step=10, reader=None,
            position_fn=SequentialCorpusPositions(),
        )
        batch = {"input_ids": np.zeros((2, 4), dtype=np.int32)}
        out = activation.maybe_inject({"step": 100}, batch)
        assert out is batch

    def test_missing_position_fn_during_active_raises(self, tmp_path):
        multi = self._build(tmp_path)
        activation = DecayPhaseActivation(
            activation_step=10, reader=multi, position_fn=None,
        )
        batch = {"input_ids": np.zeros((2, 4), dtype=np.int32)}
        with pytest.raises(RuntimeError, match="no position_fn"):
            activation.maybe_inject({"step": 10}, batch)

    def test_position_count_mismatch_raises(self, tmp_path):
        """If position_fn returns a different number than B*S we must fail loud.

        Uses in-coverage positions (so we hit the count check, not the
        out-of-coverage error from the reader).
        """
        multi = self._build(tmp_path)

        def bad_position_fn(state, batch):
            # 5 positions, all in cache coverage [0, 64); but B*S = 8.
            return np.arange(5, dtype=np.int64)

        activation = DecayPhaseActivation(
            activation_step=0, reader=multi, position_fn=bad_position_fn,
        )
        batch = {"input_ids": np.zeros((2, 4), dtype=np.int32)}  # B*S = 8
        with pytest.raises(RuntimeError, match="out of sync"):
            activation.maybe_inject({"step": 0}, batch)


# --------------------------------------------------------------------------- #
# from_yaml
# --------------------------------------------------------------------------- #
class TestFromYaml:
    def test_activation_fraction_resolves_to_step(self, tmp_path):
        yaml_path = tmp_path / "decay.yaml"
        yaml_path.write_text("activation_fraction: 0.85\nalpha: 0.3\n")
        a = DecayPhaseActivation.from_yaml(str(yaml_path), total_steps=1_000_000, reader=None)
        assert a.activation_step == 850_000

    def test_default_activation_fraction(self, tmp_path):
        yaml_path = tmp_path / "decay.yaml"
        yaml_path.write_text("alpha: 0.3\n")  # no activation_fraction key
        a = DecayPhaseActivation.from_yaml(str(yaml_path), total_steps=10_000, reader=None)
        # Default = 0.85
        assert a.activation_step == 8500

    def test_out_of_range_fraction_rejected(self, tmp_path):
        yaml_path = tmp_path / "decay.yaml"
        yaml_path.write_text("activation_fraction: 1.5\nalpha: 0.3\n")
        with pytest.raises(ValueError, match="activation_fraction"):
            DecayPhaseActivation.from_yaml(str(yaml_path), total_steps=1_000_000, reader=None)


# --------------------------------------------------------------------------- #
# Integration: loop with decay_phase
# --------------------------------------------------------------------------- #
def test_loop_with_decay_phase_injects_at_activation_step(tmp_path):
    """End-to-end: the loop calls maybe_inject before each train step, and
    we observe teacher fields in the batch passed to the train step.

    Uses a stub train_step_fn that just records the batches it sees.
    """
    from myllm.training.checkpoint import CheckpointConfig
    from myllm.training.loop import LoopConfig, run

    multi = MultiTeacherCacheReader([
        _build_reader(tmp_path / "ta", "ta", n_tokens=80, top_k=2),
    ])
    activation = DecayPhaseActivation(
        activation_step=3,
        reader=multi,
        position_fn=SequentialCorpusPositions(),
    )

    seen_batches = []

    def stub_train_step(state, batch):
        seen_batches.append({k: (v.shape if hasattr(v, "shape") else type(v).__name__) for k, v in batch.items()})
        new_state = dict(state)
        new_state["step"] = int(state["step"]) + 1
        return new_state, {"loss": 0.0, "ce": 0.0, "z_loss": 0.0}

    initial = {
        "trainable_variables": [], "non_trainable_variables": [],
        "opt_state": [], "step": 0, "lr_recovery_multiplier": 1.0,
    }

    # Synthetic data iter: 8 batches of (B=2, S=4) — 64 tokens total.
    data_iter = [
        {"input_ids": np.zeros((2, 4), dtype=np.int32),
         "labels":    np.zeros((2, 4), dtype=np.int32)}
        for _ in range(8)
    ]
    final = run(
        stub_train_step,
        initial,
        data_iter,
        loop_config=LoopConfig(total_steps=6, checkpoint_every=1_000_000),
        checkpoint_config=CheckpointConfig(root=str(tmp_path / "ckpts"), keep_last_n=1, keep_every_n=1_000_000),
        watchdog=None,
        decay_phase=activation,
    )
    assert int(final["step"]) == 6
    # Steps 0,1,2: stable (no teacher fields)
    # Steps 3,4,5: decay (teacher fields present)
    for i, observed in enumerate(seen_batches):
        if i < 3:
            assert "teacher_topk_logits" not in observed, f"step {i} (stable) should not see teacher fields"
        else:
            assert "teacher_topk_logits" in observed, f"step {i} (decay) should see teacher fields"
            assert observed["teacher_topk_logits"] == (1, 2, 4, 2)  # T=1, B=2, S=4, K=2
            assert observed["teacher_topk_indices"] == (1, 2, 4, 2)


def test_loop_without_decay_phase_unchanged(tmp_path):
    """When decay_phase=None, the loop behaves identically to pre-R0."""
    from myllm.training.checkpoint import CheckpointConfig
    from myllm.training.loop import LoopConfig, run

    seen = []

    def stub_train_step(state, batch):
        seen.append(list(batch.keys()))
        new_state = dict(state)
        new_state["step"] = int(state["step"]) + 1
        return new_state, {"loss": 0.0}

    initial = {
        "trainable_variables": [], "non_trainable_variables": [],
        "opt_state": [], "step": 0, "lr_recovery_multiplier": 1.0,
    }
    data = [{"input_ids": np.zeros((2, 4), dtype=np.int32), "labels": np.zeros((2, 4), dtype=np.int32)} for _ in range(3)]
    run(
        stub_train_step,
        initial,
        data,
        loop_config=LoopConfig(total_steps=3, checkpoint_every=1_000_000),
        checkpoint_config=CheckpointConfig(root=str(tmp_path / "ckpts"), keep_last_n=1, keep_every_n=1_000_000),
        watchdog=None,
        decay_phase=None,  # explicit disable
    )
    for batch_keys in seen:
        assert "teacher_topk_logits" not in batch_keys
