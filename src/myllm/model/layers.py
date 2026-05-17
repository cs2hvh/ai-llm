"""Decoder-only transformer layers — Llama-style, Keras-3.

Math goes through ``keras.ops`` so the same code runs on JAX (training) or
TensorFlow (tooling/inference). On the JAX backend we additionally route
attention through ``jax.nn.dot_product_attention`` — the fused FlashAttention
kernel (cuDNN on H100/B200) — which (a) cuts attention memory 4-16× at 8k
context and (b) supports proper intra-document masking via the ``mask`` arg.
Both are dossier-2026-05-11 recommendations R2/R3.

Layers implemented:
    RMSNorm                  — root-mean-square layer norm, learnable scale
    precompute_rope_cache    — cos/sin tables for rotary position embedding
    apply_rope               — rotate Q/K halves
    GroupedQueryAttention    — multi-head attention with KV-head reduction
                                (+ optional QK-norm, + optional doc-segment mask)
    SwiGLUFFN                — SiLU-gated MLP, three projections (gate/up/down)
    DecoderBlock             — pre-norm: attn + residual, ffn + residual
"""
from __future__ import annotations

import math

import keras
from keras import ops

from myllm.model.config import ModelConfig

# Detect backend once at import. The JAX fast path uses
# ``jax.nn.dot_product_attention`` (FlashAttention via cuDNN when available);
# other backends fall back to manual matmul+softmax. The fast path is a strict
# correctness-preserving swap — same math, lower memory, faster wall-clock.
_BACKEND = keras.backend.backend()
if _BACKEND == "jax":
    import jax  # noqa: E402

    _JAX_DPA = getattr(jax.nn, "dot_product_attention", None)
else:
    _JAX_DPA = None


def _residual_init_std(base_std: float, num_layers: int, scaled: bool) -> float:
    """Llama-style scaled init for residual-stream projections.

    Output projections (``wo`` and ``w_down``) are initialised with
    ``base_std / sqrt(2 * L)`` to keep residual-stream variance bounded as
    depth grows. Used for the base 1B model; pilot keeps uniform init.
    """
    if not scaled:
        return base_std
    return base_std / math.sqrt(2.0 * num_layers)


# --------------------------------------------------------------------------- #
# RMSNorm
# --------------------------------------------------------------------------- #
class RMSNorm(keras.layers.Layer):
    """y = x / sqrt(mean(x^2) + eps) * weight."""

    def __init__(self, dim: int, eps: float = 1.0e-5, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.eps = eps

    def build(self, input_shape):
        self.weight = self.add_weight(
            name="weight",
            shape=(self.dim,),
            initializer="ones",
            trainable=True,
        )
        self.built = True

    def call(self, x):
        var = ops.mean(ops.square(x), axis=-1, keepdims=True)
        x_normed = x * ops.rsqrt(var + self.eps)
        return x_normed * self.weight

    def get_config(self):
        return {**super().get_config(), "dim": self.dim, "eps": self.eps}


# --------------------------------------------------------------------------- #
# Rotary position embedding
# --------------------------------------------------------------------------- #
def precompute_rope_cache(
    head_dim: int,
    max_seq_len: int,
    base: float = 10000.0,
    dtype: str = "float32",
) -> tuple:
    """Return (cos, sin) of shape ``[max_seq_len, head_dim // 2]``."""
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (ops.arange(0, half, dtype=dtype) * 2.0 / head_dim))
    t = ops.arange(max_seq_len, dtype=dtype)
    freqs = ops.einsum("i,j->ij", t, inv_freq)  # [seq_len, half]
    return ops.cos(freqs), ops.sin(freqs)


def apply_rope(x, cos, sin):
    """Apply RoPE to ``x`` of shape ``[batch, seq_len, n_heads, head_dim]``.

    ``cos`` and ``sin`` are ``[seq_len, head_dim // 2]``.
    """
    # Pair up adjacent dims (interleaved formulation matches Llama):
    #   x_even, x_odd → (x_even * cos − x_odd * sin), (x_even * sin + x_odd * cos)
    x1 = x[..., 0::2]  # [b, s, h, head_dim/2]
    x2 = x[..., 1::2]
    cos = cos[None, :, None, :]  # broadcast over batch and heads
    sin = sin[None, :, None, :]
    rot_1 = x1 * cos - x2 * sin
    rot_2 = x1 * sin + x2 * cos
    # Re-interleave: stack last-axis pairs and flatten back.
    out = ops.stack([rot_1, rot_2], axis=-1)
    shape = ops.shape(x)
    return ops.reshape(out, shape)


# --------------------------------------------------------------------------- #
# Grouped-Query Attention
# --------------------------------------------------------------------------- #
class GroupedQueryAttention(keras.layers.Layer):
    """Multi-head attention with grouped K/V heads (n_kv_heads ≤ n_heads).

    Q has ``num_heads`` heads of dim ``head_dim``. K and V have ``num_kv_heads``
    heads of dim ``head_dim`` and are repeated ``num_heads // num_kv_heads``
    times to match Q before the dot product. Saves ~50–75% of K/V compute and
    KV-cache memory at inference vs full multi-head.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        init_std: float = 0.02,
        residual_init_std: float | None = None,
        qk_norm: bool = False,
        norm_eps: float = 1.0e-5,
        output_mult: float = 1.0,
        logit_softcap: float | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if num_heads * head_dim != hidden_dim:
            raise ValueError(
                f"num_heads*head_dim={num_heads * head_dim} != hidden_dim={hidden_dim}"
            )
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads={num_heads} must be divisible by num_kv_heads={num_kv_heads}"
            )
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.kv_dim = num_kv_heads * head_dim
        self.n_rep = num_heads // num_kv_heads
        self.scale = 1.0 / math.sqrt(head_dim)
        self.init_std = init_std
        self.residual_init_std = residual_init_std if residual_init_std is not None else init_std
        self.qk_norm = qk_norm
        self.norm_eps = norm_eps
        # muP attention-output multiplier (1.0 = no-op, SP). When set to
        # 1/width_mult, scales the attention block's residual contribution
        # so signal magnitude is invariant to model width. See
        # `docs/mup_design.md` §"Change 1".
        self.output_mult = output_mult
        # Gemma-2-style soft-cap on attention QK logits:
        #   scores = softcap * tanh(scores / softcap)
        # Bounds extreme attention scores so the softmax can't saturate on
        # pathological inputs (e.g. repetition / low-entropy spans). None =
        # disabled (lets the cuDNN-FlashAttention fast path stay in use).
        # Setting any non-None value forces the manual attention path
        # because jax.nn.dot_product_attention (JAX 0.4.38) does not expose
        # logits_soft_cap on the cuDNN backend. ~5-10% wall-time tax at
        # seq=8192; budget vs production-default OFF.
        self.logit_softcap = logit_softcap

    def build(self, input_shape):
        init = keras.initializers.RandomNormal(stddev=self.init_std)
        out_init = keras.initializers.RandomNormal(stddev=self.residual_init_std)
        self.wq = self.add_weight(
            name="wq", shape=(self.hidden_dim, self.hidden_dim), initializer=init
        )
        self.wk = self.add_weight(
            name="wk", shape=(self.hidden_dim, self.kv_dim), initializer=init
        )
        self.wv = self.add_weight(
            name="wv", shape=(self.hidden_dim, self.kv_dim), initializer=init
        )
        self.wo = self.add_weight(
            name="wo", shape=(self.hidden_dim, self.hidden_dim), initializer=out_init
        )
        if self.qk_norm:
            # Per-head RMSNorm applied post-RoPE, pre-attention. Bounds the
            # magnitudes of Q and K so the softmax doesn't saturate at long
            # training horizons. Consensus default at 1B+ as of 2026 (Gemini,
            # DeepSeek-V3, Llama-3 70B). ~0.3% FLOPs overhead, near-zero risk.
            self.q_norm = RMSNorm(self.head_dim, self.norm_eps, name="q_norm")
            self.k_norm = RMSNorm(self.head_dim, self.norm_eps, name="k_norm")
        self.built = True

    def call(self, x, cos, sin, mask=None, segment_ids=None):
        """Apply attention.

        Args:
            x:           ``[batch, seq, hidden]`` input.
            cos, sin:    RoPE caches, ``[seq, head_dim/2]``.
            mask:        optional additive attention mask ``[*, *, seq, seq]``
                         where ``-inf`` blocks and ``0`` permits. Used as-is
                         on the manual path; on the JAX fast path it is
                         converted to a boolean mask (entries above ``-1e8``
                         are treated as permitted).
            segment_ids: optional ``[batch, seq]`` integer document IDs. When
                         provided, the layer builds a combined causal +
                         intra-document mask, OVERRIDING ``mask``. This is the
                         R2 doc-mask path — required for packed multi-doc
                         training shards to be correct (not just faster).
        """
        # x: [batch, seq, hidden]
        b = ops.shape(x)[0]
        s = ops.shape(x)[1]

        q = ops.matmul(x, self.wq)
        k = ops.matmul(x, self.wk)
        v = ops.matmul(x, self.wv)

        q = ops.reshape(q, (b, s, self.num_heads, self.head_dim))
        k = ops.reshape(k, (b, s, self.num_kv_heads, self.head_dim))
        v = ops.reshape(v, (b, s, self.num_kv_heads, self.head_dim))

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # R3: QK-norm, post-RoPE per Llama-3 convention.
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # GQA: repeat K, V along the head axis to match Q.
        # (jax.nn.dot_product_attention also supports head-broadcasting
        # natively on the cuDNN impl, but explicit repeat keeps the manual
        # path correct and matches XLA-fallback expectations.)
        if self.n_rep > 1:
            k = ops.repeat(k, self.n_rep, axis=2)
            v = ops.repeat(v, self.n_rep, axis=2)

        # R2: build the effective attention mask.
        #
        #   - segment_ids given → bool mask of (same_segment AND causal)
        #   - mask given        → use additive mask as-is (legacy path)
        #   - neither           → pure causal
        bool_mask = None
        if segment_ids is not None:
            same_seg = ops.equal(
                segment_ids[..., :, None], segment_ids[..., None, :]
            )                                              # [b, s, s]
            causal_b = ops.cast(ops.tri(s), "bool")        # [s, s]
            bool_mask = ops.logical_and(
                same_seg, causal_b[None, :, :]
            )[:, None, :, :]                               # [b, 1, s, s]

        # When logit_softcap is set, force the manual path: cuDNN FlashAttention
        # via jax.nn.dot_product_attention (JAX 0.4.38) does NOT expose
        # logits_soft_cap. The manual path applies softcap directly on scores
        # before the mask + softmax. Slower at long seq but mathematically
        # equivalent to Gemma 2's softcap-FA kernel.
        use_fast_path = _JAX_DPA is not None and self.logit_softcap is None

        if use_fast_path:
            # JAX fast path: fused FlashAttention via jax.nn.dot_product_attention.
            # Input format is [B, S, N, H]; no transpose needed.
            if bool_mask is not None:
                out = _JAX_DPA(q, k, v, mask=bool_mask, scale=self.scale)
            elif mask is not None:
                # Convert additive (-inf=blocked, 0=allowed) to boolean.
                bool_from_add = ops.greater(mask, ops.cast(-1.0e8, mask.dtype))
                out = _JAX_DPA(q, k, v, mask=bool_from_add, scale=self.scale)
            else:
                out = _JAX_DPA(q, k, v, is_causal=True, scale=self.scale)
            # out: [b, s, num_heads, head_dim]
        else:
            # Manual fallback (works on any backend; same math). Also taken
            # whenever logit_softcap is set.
            q = ops.transpose(q, (0, 2, 1, 3))
            k = ops.transpose(k, (0, 2, 1, 3))
            v = ops.transpose(v, (0, 2, 1, 3))
            scores = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * self.scale
            if self.logit_softcap is not None:
                # Gemma 2 softcap: scores = c * tanh(scores / c). Squashes
                # extreme values into [-c, +c] before the mask + softmax.
                cap = ops.cast(self.logit_softcap, scores.dtype)
                scores = cap * ops.tanh(scores / cap)
            if bool_mask is not None:
                add_mask = ops.where(
                    bool_mask,
                    ops.cast(0.0, scores.dtype),
                    ops.cast(-1.0e9, scores.dtype),
                )
                scores = scores + add_mask
            elif mask is not None:
                scores = scores + mask
            else:
                # Fall back to a causal-only mask so we don't accidentally
                # train acausally if both kwargs are None.
                i = ops.arange(s)[:, None]
                j = ops.arange(s)[None, :]
                add_mask = ops.where(
                    j > i,
                    ops.cast(-1.0e9, scores.dtype),
                    ops.cast(0.0, scores.dtype),
                )
                scores = scores + add_mask[None, None, :, :]
            attn = ops.softmax(scores, axis=-1)
            out = ops.matmul(attn, v)                       # [b, h, s, d]
            out = ops.transpose(out, (0, 2, 1, 3))          # [b, s, h, d]

        out = ops.reshape(out, (b, s, self.hidden_dim))
        out = ops.matmul(out, self.wo)
        if self.output_mult != 1.0:
            out = out * self.output_mult
        return out

    def get_config(self):
        return {
            **super().get_config(),
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "init_std": self.init_std,
            "qk_norm": self.qk_norm,
            "norm_eps": self.norm_eps,
            "output_mult": self.output_mult,
            "logit_softcap": self.logit_softcap,
        }


# --------------------------------------------------------------------------- #
# SwiGLU FFN
# --------------------------------------------------------------------------- #
class SwiGLUFFN(keras.layers.Layer):
    """y = down(silu(gate(x)) * up(x))."""

    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        init_std: float = 0.02,
        residual_init_std: float | None = None,
        output_mult: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.init_std = init_std
        self.residual_init_std = residual_init_std if residual_init_std is not None else init_std
        # muP FFN-output multiplier — see GroupedQueryAttention.output_mult.
        self.output_mult = output_mult

    def build(self, input_shape):
        init = keras.initializers.RandomNormal(stddev=self.init_std)
        out_init = keras.initializers.RandomNormal(stddev=self.residual_init_std)
        self.w_gate = self.add_weight(
            name="w_gate", shape=(self.hidden_dim, self.ffn_dim), initializer=init
        )
        self.w_up = self.add_weight(
            name="w_up", shape=(self.hidden_dim, self.ffn_dim), initializer=init
        )
        self.w_down = self.add_weight(
            name="w_down", shape=(self.ffn_dim, self.hidden_dim), initializer=out_init
        )
        self.built = True

    def call(self, x):
        gate = ops.silu(ops.matmul(x, self.w_gate))
        up = ops.matmul(x, self.w_up)
        out = ops.matmul(gate * up, self.w_down)
        if self.output_mult != 1.0:
            out = out * self.output_mult
        return out

    def get_config(self):
        return {
            **super().get_config(),
            "hidden_dim": self.hidden_dim,
            "ffn_dim": self.ffn_dim,
            "init_std": self.init_std,
            "output_mult": self.output_mult,
        }


# --------------------------------------------------------------------------- #
# Decoder block
# --------------------------------------------------------------------------- #
class DecoderBlock(keras.layers.Layer):
    """Pre-norm transformer block: ``x = x + attn(norm(x)); x = x + ffn(norm(x))``."""

    def __init__(self, config: ModelConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        residual_std = _residual_init_std(
            config.init_std, config.layers, config.scaled_init_for_residuals
        )
        # muP output multipliers: 1/width_mult when muP is enabled,
        # otherwise no-op (1.0). Each gated by the matching MupConfig flag
        # so we can ablate individual changes.
        width_mult = config.mup_width_multiplier()
        attn_out_mult = (
            1.0 / width_mult
            if (config.mup is not None and config.mup.apply_attn_output_mult)
            else 1.0
        )
        ffn_out_mult = (
            1.0 / width_mult
            if (config.mup is not None and config.mup.apply_ffn_output_mult)
            else 1.0
        )
        self.attn_norm = RMSNorm(config.hidden_dim, config.norm_eps, name="attn_norm")
        self.attn = GroupedQueryAttention(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            init_std=config.init_std,
            residual_init_std=residual_std,
            qk_norm=config.qk_norm,
            norm_eps=config.norm_eps,
            output_mult=attn_out_mult,
            logit_softcap=config.attn_logit_softcap,
            name="attn",
        )
        self.ffn_norm = RMSNorm(config.hidden_dim, config.norm_eps, name="ffn_norm")
        self.ffn = SwiGLUFFN(
            hidden_dim=config.hidden_dim,
            ffn_dim=config.ffn_dim,
            init_std=config.init_std,
            residual_init_std=residual_std,
            output_mult=ffn_out_mult,
            name="ffn",
        )

    def call(self, x, cos, sin, mask=None, segment_ids=None):
        x = x + self.attn(
            self.attn_norm(x), cos, sin, mask=mask, segment_ids=segment_ids
        )
        x = x + self.ffn(self.ffn_norm(x))
        return x
