"""muP optimizer regression tests — R1 part 2.

Validates the per-parameter LR scaling added to ``build_optimizer``:

  1. Variable-path classification: ``embed`` paths → embedding,
     ``norm`` paths → norm, everything else → hidden.

  2. Default-off invariance: calling ``build_optimizer(config, lr_fn)``
     without muP kwargs produces a single-AdamW chain that yields the
     same updates as the original implementation (backwards compat).

  3. muP path: with ``mup_width_mult=m`` and proper labels, hidden-group
     updates are scaled by ``1/m`` relative to embedding/norm updates.

  4. Input validation: bad labels raise; invalid ``mup_width_mult`` raises.

We test the optimizer mechanics directly (no model required) by hand-
building a parameter PyTree, gradients, and verifying the produced
updates have the expected per-group magnitudes.
"""
from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
optax = pytest.importorskip("optax")
np = pytest.importorskip("numpy")

from myllm.training.optimizer import (  # noqa: E402
    PARAM_GROUP_EMBEDDING,
    PARAM_GROUP_HIDDEN,
    PARAM_GROUP_NORM,
    OptimizerConfig,
    build_optimizer,
    label_model_variables,
    label_variable_for_mup,
)


# --------------------------------------------------------------------------- #
# Variable-path classification
# --------------------------------------------------------------------------- #
class TestLabelVariableForMup:
    def test_embedding_paths(self):
        assert label_variable_for_mup("transformer_lm/tok_embed/embeddings") == PARAM_GROUP_EMBEDDING
        assert label_variable_for_mup("tok_embed/embeddings") == PARAM_GROUP_EMBEDDING
        # Untied LM head also unscaled in muP recipe.
        assert label_variable_for_mup("transformer_lm/lm_head/kernel") == PARAM_GROUP_EMBEDDING

    def test_norm_paths(self):
        assert label_variable_for_mup("transformer_lm/block_0/attn_norm/weight") == PARAM_GROUP_NORM
        assert label_variable_for_mup("block_5/ffn_norm/weight") == PARAM_GROUP_NORM
        assert label_variable_for_mup("transformer_lm/final_norm/weight") == PARAM_GROUP_NORM
        # QK-norm (added in R3) — also norm group.
        assert label_variable_for_mup("block_0/attn/q_norm/weight") == PARAM_GROUP_NORM
        assert label_variable_for_mup("block_0/attn/k_norm/weight") == PARAM_GROUP_NORM

    def test_hidden_paths(self):
        for path in [
            "block_0/attn/wq",
            "block_0/attn/wk",
            "block_0/attn/wv",
            "block_0/attn/wo",
            "block_3/ffn/w_gate",
            "block_3/ffn/w_up",
            "block_3/ffn/w_down",
        ]:
            assert label_variable_for_mup(path) == PARAM_GROUP_HIDDEN, path

    def test_case_insensitive(self):
        assert label_variable_for_mup("TOK_EMBED/Embeddings") == PARAM_GROUP_EMBEDDING
        assert label_variable_for_mup("Block_0/Attn_Norm/Weight") == PARAM_GROUP_NORM

    def test_empty_path_defaults_to_hidden(self):
        # Conservative fallback — unknown variables get LR-scaled at base.
        # This is the safer default than "embedding" because we never want
        # to accidentally not-scale a real hidden weight.
        assert label_variable_for_mup("") == PARAM_GROUP_HIDDEN
        assert label_variable_for_mup(None) == PARAM_GROUP_HIDDEN


class TestLabelModelVariables:
    def test_labels_match_variable_order(self):
        """Each label must correspond to the variable at the same index."""

        # Build a fake model-like object with a few named "variables".
        class _FakeVar:
            def __init__(self, path):
                self.path = path
                self.name = path

        class _FakeModel:
            trainable_variables = [
                _FakeVar("tok_embed/embeddings"),
                _FakeVar("block_0/attn_norm/weight"),
                _FakeVar("block_0/attn/wq"),
                _FakeVar("block_0/ffn/w_gate"),
                _FakeVar("final_norm/weight"),
            ]

        labels = label_model_variables(_FakeModel())
        assert labels == [
            PARAM_GROUP_EMBEDDING,
            PARAM_GROUP_NORM,
            PARAM_GROUP_HIDDEN,
            PARAM_GROUP_HIDDEN,
            PARAM_GROUP_NORM,
        ]


# --------------------------------------------------------------------------- #
# Default-off invariance
# --------------------------------------------------------------------------- #
def _const_lr(_step):
    return 1.0e-3


def _toy_params():
    """A flat list of three params — embedding-like, norm-like, hidden-like.

    All start at zero so weight-decay contributions are zero too — we want
    to isolate the LR-scaling effect, not measure weight-decay differences.
    """
    return [
        jnp.zeros((4, 8)),   # "embedding"
        jnp.zeros((4,)),     # "norm" (zero, not ones — see docstring)
        jnp.zeros((4, 4)),   # "hidden"
    ]


def _toy_grads():
    """Unit gradients of the same shape as _toy_params."""
    return [
        jnp.ones((4, 8)),
        jnp.ones((4,)),
        jnp.ones((4, 4)),
    ]


def test_default_off_matches_legacy_optimizer():
    """build_optimizer with no muP kwargs should produce the same updates
    as the legacy single-AdamW chain.

    We assert this by checking the produced updates have a single
    consistent magnitude across all param groups (i.e. NOT per-group
    scaled).
    """
    # weight_decay=0 to isolate the LR-scaling effect from AdamW's decay term.
    cfg = OptimizerConfig(peak_lr=1.0e-3, weight_decay=0.0)
    opt = build_optimizer(cfg, _const_lr)

    params = _toy_params()
    opt_state = opt.init(params)
    updates, _ = opt.update(_toy_grads(), opt_state, params)

    # All updates should have the same absolute magnitude (after AdamW
    # preconditioning of unit gradients with zero initial moments, every
    # element should be exactly -lr).
    magnitudes = [float(jnp.mean(jnp.abs(u))) for u in updates]
    assert max(magnitudes) - min(magnitudes) < 1.0e-6, (
        f"default-off optimizer should produce uniform updates, got {magnitudes}"
    )


# --------------------------------------------------------------------------- #
# muP path — hidden updates scaled by 1/width_mult
# --------------------------------------------------------------------------- #
def test_mup_optimizer_scales_hidden_updates_by_1_over_width_mult():
    """With width_mult=2.0 and proper labels, hidden updates should be
    half the magnitude of embedding/norm updates.
    """
    # weight_decay=0 to isolate the LR-scaling effect from AdamW's decay term.
    cfg = OptimizerConfig(peak_lr=1.0e-3, weight_decay=0.0)
    labels = [PARAM_GROUP_EMBEDDING, PARAM_GROUP_NORM, PARAM_GROUP_HIDDEN]
    opt = build_optimizer(cfg, _const_lr, param_labels=labels, mup_width_mult=2.0)

    params = _toy_params()
    opt_state = opt.init(params)
    updates, _ = opt.update(_toy_grads(), opt_state, params)

    embed_mag = float(jnp.mean(jnp.abs(updates[0])))
    norm_mag = float(jnp.mean(jnp.abs(updates[1])))
    hidden_mag = float(jnp.mean(jnp.abs(updates[2])))

    # embedding ≈ norm (both get base LR).
    assert abs(embed_mag - norm_mag) < 1.0e-6, (
        f"embedding and norm updates should be equal, got {embed_mag} vs {norm_mag}"
    )
    # hidden = embedding / 2.
    ratio = hidden_mag / embed_mag
    assert ratio == pytest.approx(0.5, rel=1.0e-4), (
        f"hidden updates should be 0.5× embedding, got ratio {ratio:.6f}"
    )


def test_mup_optimizer_scales_correctly_at_width_mult_4():
    """At width_mult=4.0, hidden updates should be 1/4 the magnitude."""
    # weight_decay=0 to isolate the LR-scaling effect from AdamW's decay term.
    cfg = OptimizerConfig(peak_lr=1.0e-3, weight_decay=0.0)
    labels = [PARAM_GROUP_EMBEDDING, PARAM_GROUP_NORM, PARAM_GROUP_HIDDEN]
    opt = build_optimizer(cfg, _const_lr, param_labels=labels, mup_width_mult=4.0)
    params = _toy_params()
    opt_state = opt.init(params)
    updates, _ = opt.update(_toy_grads(), opt_state, params)

    ratio = float(jnp.mean(jnp.abs(updates[2]))) / float(jnp.mean(jnp.abs(updates[0])))
    assert ratio == pytest.approx(0.25, rel=1.0e-4), (
        f"hidden updates should be 0.25× embedding at width_mult=4, got {ratio:.6f}"
    )


def test_mup_width_mult_1_with_labels_matches_default():
    """Even with labels provided, width_mult=1.0 should collapse to the
    default-off optimizer (no per-group scaling)."""
    # weight_decay=0 to isolate the LR-scaling effect from AdamW's decay term.
    cfg = OptimizerConfig(peak_lr=1.0e-3, weight_decay=0.0)
    labels = [PARAM_GROUP_EMBEDDING, PARAM_GROUP_NORM, PARAM_GROUP_HIDDEN]
    opt_default = build_optimizer(cfg, _const_lr)
    opt_mup = build_optimizer(cfg, _const_lr, param_labels=labels, mup_width_mult=1.0)

    params = _toy_params()
    state_default = opt_default.init(params)
    state_mup = opt_mup.init(params)
    updates_default, _ = opt_default.update(_toy_grads(), state_default, params)
    updates_mup, _ = opt_mup.update(_toy_grads(), state_mup, params)

    for u_d, u_m in zip(updates_default, updates_mup):
        diff = float(jnp.max(jnp.abs(u_d - u_m)))
        assert diff < 1.0e-6, (
            f"width_mult=1.0 should match default-off; got max diff {diff}"
        )


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #
def test_invalid_param_label_raises():
    cfg = OptimizerConfig()
    with pytest.raises(ValueError, match="unknown groups"):
        build_optimizer(
            cfg, _const_lr,
            param_labels=["embedding", "MYSTERY_GROUP"],
            mup_width_mult=2.0,
        )


def test_invalid_width_mult_raises():
    cfg = OptimizerConfig()
    with pytest.raises(ValueError, match="mup_width_mult"):
        build_optimizer(
            cfg, _const_lr,
            param_labels=["hidden"],
            mup_width_mult=0.0,
        )
    with pytest.raises(ValueError, match="mup_width_mult"):
        build_optimizer(
            cfg, _const_lr,
            param_labels=["hidden"],
            mup_width_mult=-1.0,
        )
