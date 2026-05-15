"""Tests for the forward-only eval step (Phase 1.5).

The training-time eval hook had been re-using train_step_fn for eval —
that works without FSDP but breaks under FSDP because train_step uses
``donate_argnums=(0,)`` which destroys the input state buffer in-place.
This separate eval_step is forward-only: no grads, no opt update, no
donation. State is invariant across calls.

Invariants pinned here:
  1. State is NOT mutated. The exact same trainable_variables come back
     after eval_step(state, batch) returns.
  2. The reported `loss` (ce + z_loss_coef * z_loss) matches the loss
     train_step's forward computes on the same batch + state — within
     numerical noise.
  3. The FSDP path (state_shardings provided) returns a finite loss
     under a 4-device CPU mesh, matching the non-FSDP path within
     numerical noise (the L2 parity invariant from train_step_fsdp).
  4. When return_per_token_nll=True, metrics includes nll_per_token:
     [B, S] and weight_per_token: [B, S]. Used by Phase 1.2 per-source
     val loss to bucket tokens by source_id.
  5. The chunked-CE eval path matches the non-chunked path within
     numerical noise (locks the chunked branch in train_step_chunked_ce).
"""
from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import numpy as np
import pytest

# Force-import keras before model so backend selection is locked.
import keras  # noqa: F401, E402

from myllm.model.config import ModelConfig  # noqa: E402
from myllm.model.transformer import build_model  # noqa: E402
from myllm.training.eval_step import make_eval_step  # noqa: E402
from myllm.training.train_step import make_train_step  # noqa: E402


def _tiny_cfg(vocab_size: int = 64) -> ModelConfig:
    return ModelConfig(
        name="tiny_eval_test",
        arch="llama_decoder",
        layers=2,
        hidden_dim=32,
        ffn_dim=64,
        num_heads=4,
        num_kv_heads=2,
        head_dim=8,
        vocab_size=vocab_size,
        tie_embeddings=True,
        context_length=16,
        position="rope",
        rope_base=10000.0,
        norm="rmsnorm",
        norm_eps=1e-5,
        activation="swiglu",
        init_std=0.02,
        scaled_init_for_residuals=False,
        z_loss_coef=1e-4,
        qk_norm=False,
        gradient_checkpointing=False,
    )


def _initial_state(model, optimizer):
    import jax.numpy as jnp
    trainable = [v.value for v in model.trainable_variables]
    non_trainable = [v.value for v in model.non_trainable_variables]
    return {
        "trainable_variables": trainable,
        "non_trainable_variables": non_trainable,
        "opt_state": optimizer.init(trainable),
        "step": jnp.array(0, dtype=jnp.int32),
        "lr_recovery_multiplier": jnp.array(1.0, dtype=jnp.float32),
    }


def _make_batch(seed: int, vocab: int, B: int = 4, S: int = 4):
    rng = np.random.default_rng(seed)
    return {
        "input_ids": rng.integers(0, vocab, size=(B, S)).astype(np.int32),
        "labels": rng.integers(0, vocab, size=(B, S)).astype(np.int32),
    }


def _mesh(data_size: int = 4):
    import jax
    from jax.sharding import Mesh
    devices = jax.devices()
    if len(devices) < data_size:
        pytest.skip(f"need {data_size} CPU devices via XLA_FLAGS")
    arr = np.asarray(devices[:data_size]).reshape(data_size, 1)
    return Mesh(arr, axis_names=("data", "model"))


class TestEvalStepStateInvariance:
    """The key invariant: eval_step must not mutate state."""

    def test_state_not_mutated_after_eval(self):
        try:
            import optax
        except ImportError:
            pytest.skip("optax not installed")
        import jax.numpy as jnp

        vocab = 64
        cfg = _tiny_cfg(vocab_size=vocab)
        model = build_model(cfg)
        optimizer = optax.adamw(learning_rate=1e-3)
        batch = _make_batch(seed=42, vocab=vocab)

        state = _initial_state(model, optimizer)
        # Snapshot every trainable leaf BEFORE the call.
        before = [np.asarray(t) for t in state["trainable_variables"]]
        before_step = int(state["step"])

        eval_fn = make_eval_step(model, z_loss_coef=1e-4)
        metrics = eval_fn(state, batch)

        assert jnp.isfinite(metrics["loss"]), "loss must be finite"
        # After must equal before, leafwise — no in-place mutation, no
        # accidental optimizer step.
        after = [np.asarray(t) for t in state["trainable_variables"]]
        assert len(before) == len(after)
        for i, (b, a) in enumerate(zip(before, after)):
            assert b.shape == a.shape
            assert np.array_equal(b, a), (
                f"trainable leaf {i} mutated by eval_step "
                f"(max |Δ|={np.max(np.abs(b - a))})"
            )
        # The step counter is just data — eval shouldn't move it.
        assert int(state["step"]) == before_step


class TestEvalStepLossParityWithTrainStep:
    """The reported loss must match the forward computed by train_step."""

    def test_eval_loss_matches_train_step_forward(self):
        # Same model, same batch, same state -> same loss value. The
        # train_step does grad + opt update on top; the loss it REPORTS
        # for the current batch is computed BEFORE that update, so it
        # must equal the eval_step's loss.
        try:
            import optax
        except ImportError:
            pytest.skip("optax not installed")
        import jax.numpy as jnp

        vocab = 64
        cfg = _tiny_cfg(vocab_size=vocab)
        model = build_model(cfg)
        optimizer = optax.adamw(learning_rate=1e-3)
        batch = _make_batch(seed=7, vocab=vocab)
        state = _initial_state(model, optimizer)

        train_fn = make_train_step(model, optimizer, z_loss_coef=1e-4)
        eval_fn = make_eval_step(model, z_loss_coef=1e-4)

        # Train step reports the current-batch loss; we compare against
        # the eval step's loss on the SAME pre-update state.
        _new_state, train_metrics = train_fn(state, batch)
        eval_metrics = eval_fn(state, batch)

        train_loss = float(train_metrics["loss"])
        eval_loss = float(eval_metrics["loss"])
        # bf16 mixed-precision noise is on the order of 1e-3 even on
        # tiny models. Loosen the tolerance accordingly.
        assert jnp.isfinite(train_loss) and jnp.isfinite(eval_loss)
        assert abs(train_loss - eval_loss) < 1e-3, (
            f"train_loss={train_loss}, eval_loss={eval_loss}; "
            f"|Δ|={abs(train_loss - eval_loss)} exceeds bf16 noise floor"
        )


class TestEvalStepPerTokenNLL:
    """When return_per_token_nll=True, metrics expose nll_per_token + weight_per_token."""

    def test_per_token_nll_shape_and_finite(self):
        try:
            import optax
        except ImportError:
            pytest.skip("optax not installed")

        vocab = 64
        cfg = _tiny_cfg(vocab_size=vocab)
        model = build_model(cfg)
        optimizer = optax.adamw(learning_rate=1e-3)
        batch = _make_batch(seed=11, vocab=vocab, B=4, S=4)

        state = _initial_state(model, optimizer)
        eval_fn = make_eval_step(model, z_loss_coef=1e-4, return_per_token_nll=True)
        metrics = eval_fn(state, batch)

        # Per-token NLL must have [B, S] shape — that's what enables
        # per-source bucketing in Phase 1.2.
        nll = np.asarray(metrics["nll_per_token"])
        w = np.asarray(metrics["weight_per_token"])
        assert nll.shape == (4, 4), f"unexpected nll shape: {nll.shape}"
        assert w.shape == (4, 4), f"unexpected weight shape: {w.shape}"
        assert np.all(np.isfinite(nll)), "nll_per_token must be all-finite"
        # No mask was passed -> weight_per_token is all-ones.
        assert np.allclose(w, 1.0), "weight_per_token should be 1s when no mask"

    def test_per_token_nll_mean_matches_ce(self):
        # The MEAN of nll_per_token must match the reported `ce` metric
        # (within float noise). This pins the contract that downstream
        # per-source aggregation gives the same total as the unbucketed
        # ce when bucketed back over all sources.
        try:
            import optax
        except ImportError:
            pytest.skip("optax not installed")

        vocab = 64
        cfg = _tiny_cfg(vocab_size=vocab)
        model = build_model(cfg)
        optimizer = optax.adamw(learning_rate=1e-3)
        batch = _make_batch(seed=13, vocab=vocab, B=4, S=4)
        state = _initial_state(model, optimizer)
        eval_fn = make_eval_step(model, z_loss_coef=1e-4, return_per_token_nll=True)
        metrics = eval_fn(state, batch)

        nll = np.asarray(metrics["nll_per_token"])
        ce = float(metrics["ce"])
        # No mask -> mean of nll == ce (within bf16 noise).
        assert abs(float(nll.mean()) - ce) < 1e-3


class TestEvalStepFSDP:
    """FSDP path: state_shardings declared; no donation; loss matches no-FSDP."""

    def test_fsdp_eval_step_loss_parity(self):
        try:
            import optax
        except ImportError:
            pytest.skip("optax not installed")
        import jax
        from jax.sharding import NamedSharding, PartitionSpec as P
        from myllm.training.mesh import make_param_shardings
        from myllm.training.optimizer import make_optimizer_state_sharding

        vocab = 64
        cfg = _tiny_cfg(vocab_size=vocab)
        model = build_model(cfg)
        optimizer = optax.adamw(learning_rate=1e-3)
        mesh = _mesh(4)
        batch = _make_batch(seed=23, vocab=vocab, B=4, S=4)

        # --- Path A: no FSDP ---
        eval_a = make_eval_step(model, z_loss_coef=1e-4)
        state_a = _initial_state(model, optimizer)
        metrics_a = eval_a(state_a, batch)

        # --- Path B: FSDP ---
        state_b = _initial_state(model, optimizer)
        replicate = NamedSharding(mesh, P())
        state_shardings = {
            "trainable_variables": make_param_shardings(state_b["trainable_variables"], mesh),
            "non_trainable_variables": make_param_shardings(state_b["non_trainable_variables"], mesh),
            "opt_state": make_optimizer_state_sharding(optimizer, state_b["trainable_variables"], mesh),
            "step": replicate,
            "lr_recovery_multiplier": replicate,
        }
        batch_sharding = NamedSharding(mesh, P("data"))
        state_b = jax.tree.map(
            lambda x, s: jax.device_put(x, s), state_b, state_shardings,
        )
        batch_b = {k: jax.device_put(v, batch_sharding) for k, v in batch.items()}
        eval_b = make_eval_step(
            model, z_loss_coef=1e-4,
            state_shardings=state_shardings, batch_sharding=batch_sharding,
        )
        metrics_b = eval_b(state_b, batch_b)

        loss_a = float(metrics_a["loss"])
        loss_b = float(metrics_b["loss"])
        assert abs(loss_a - loss_b) < 1e-3, (
            f"FSDP eval_loss={loss_b} drifted from no-FSDP={loss_a}; |Δ|={abs(loss_a - loss_b)}"
        )

    def test_fsdp_eval_step_no_donation_state_reusable(self):
        # After an FSDP eval call, the input state object must still be
        # usable for another call (i.e., its buffers weren't donated).
        # If donate_argnums were set, the second call would fail with
        # "Argument was previously donated".
        try:
            import optax
        except ImportError:
            pytest.skip("optax not installed")
        import jax
        from jax.sharding import NamedSharding, PartitionSpec as P
        from myllm.training.mesh import make_param_shardings
        from myllm.training.optimizer import make_optimizer_state_sharding

        vocab = 64
        cfg = _tiny_cfg(vocab_size=vocab)
        model = build_model(cfg)
        optimizer = optax.adamw(learning_rate=1e-3)
        mesh = _mesh(4)
        batch = _make_batch(seed=31, vocab=vocab, B=4, S=4)

        state = _initial_state(model, optimizer)
        replicate = NamedSharding(mesh, P())
        state_shardings = {
            "trainable_variables": make_param_shardings(state["trainable_variables"], mesh),
            "non_trainable_variables": make_param_shardings(state["non_trainable_variables"], mesh),
            "opt_state": make_optimizer_state_sharding(optimizer, state["trainable_variables"], mesh),
            "step": replicate,
            "lr_recovery_multiplier": replicate,
        }
        batch_sharding = NamedSharding(mesh, P("data"))
        state = jax.tree.map(
            lambda x, s: jax.device_put(x, s), state, state_shardings,
        )
        batch_placed = {k: jax.device_put(v, batch_sharding) for k, v in batch.items()}
        eval_fn = make_eval_step(
            model, z_loss_coef=1e-4,
            state_shardings=state_shardings, batch_sharding=batch_sharding,
        )
        # Two consecutive calls must both work. If the first donated state,
        # the second would raise.
        m1 = eval_fn(state, batch_placed)
        m2 = eval_fn(state, batch_placed)
        # Same state + same batch -> same loss. Locked.
        assert abs(float(m1["loss"]) - float(m2["loss"])) < 1e-6


class TestEvalStepChunkedCE:
    """Chunked-CE eval path matches the non-chunked path within bf16 noise."""

    def test_chunked_and_full_eval_paths_match(self):
        # vocab_size must be divisible by num_chunks. 64 / 4 = 16.
        try:
            import optax
        except ImportError:
            pytest.skip("optax not installed")

        vocab = 64
        cfg = _tiny_cfg(vocab_size=vocab)
        model = build_model(cfg)
        optimizer = optax.adamw(learning_rate=1e-3)
        batch = _make_batch(seed=37, vocab=vocab, B=4, S=4)
        state = _initial_state(model, optimizer)

        full = make_eval_step(model, z_loss_coef=1e-4, use_chunked_ce=False)
        chunked = make_eval_step(
            model, z_loss_coef=1e-4,
            use_chunked_ce=True, chunked_ce_num_chunks=4,
        )

        loss_full = float(full(state, batch)["loss"])
        loss_chunked = float(chunked(state, batch)["loss"])
        # Same numerical tolerance as test_train_step_chunked_ce.
        assert abs(loss_full - loss_chunked) < 1e-3, (
            f"chunked={loss_chunked}, full={loss_full}; "
            f"|Δ|={abs(loss_full - loss_chunked)}"
        )
