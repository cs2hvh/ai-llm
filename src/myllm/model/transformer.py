"""Full TransformerLM assembly: token embed → N decoder blocks → norm → lm_head."""
from __future__ import annotations

import keras
from keras import ops

from myllm.model.config import ModelConfig
from myllm.model.layers import DecoderBlock, RMSNorm, precompute_rope_cache

# Optional jax import for activation checkpointing. The model still runs
# correctly without it (e.g., TF backend or older JAX); gradient checkpointing
# just becomes a no-op in that case.
try:
    import jax  # noqa: F401
    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False


def causal_mask(seq_len: int, dtype: str = "float32"):
    """Return additive causal mask of shape ``[1, 1, seq_len, seq_len]``.

    Upper-triangular positions get ``-inf``; lower-triangular get 0.
    """
    # Build via subtraction so it works under JIT in all backends.
    i = ops.arange(seq_len)[:, None]
    j = ops.arange(seq_len)[None, :]
    mask = ops.where(j > i, ops.cast(-1.0e9, dtype), ops.cast(0.0, dtype))
    return mask[None, None, :, :]


class TransformerLM(keras.Model):
    """Decoder-only language model. Returns logits over the vocabulary."""

    def __init__(self, config: ModelConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config

        self.embed = keras.layers.Embedding(
            input_dim=config.vocab_size,
            output_dim=config.hidden_dim,
            embeddings_initializer=keras.initializers.RandomNormal(stddev=config.init_std),
            name="tok_embed",
        )
        self.blocks = [
            DecoderBlock(config, name=f"block_{i}") for i in range(config.layers)
        ]
        self.final_norm = RMSNorm(config.hidden_dim, config.norm_eps, name="final_norm")
        if not config.tie_embeddings:
            self.lm_head = keras.layers.Dense(
                config.vocab_size,
                use_bias=False,
                kernel_initializer=keras.initializers.RandomNormal(stddev=config.init_std),
                name="lm_head",
            )

    def build(self, input_shape):
        # Precompute RoPE tables as non-trainable buffers.
        cos, sin = precompute_rope_cache(
            head_dim=self.config.head_dim,
            max_seq_len=self.config.context_length,
            base=self.config.rope_base,
        )
        self._rope_cos = keras.Variable(cos, trainable=False, name="rope_cos")
        self._rope_sin = keras.Variable(sin, trainable=False, name="rope_sin")
        self.built = True

        # 2026-05-12 (gradient checkpointing): force all sublayer Variables
        # to materialise via a NON-CHECKPOINTED dummy forward. Without this
        # warm-up, the very first real call() would lazy-init RMSNorm.scale /
        # other Variables INSIDE the jax.checkpoint-traced function, causing
        # an UnexpectedTracerError ("intermediate value escaped the trace
        # scope"). Setting _building=True signals call() to skip the
        # checkpoint wrap for this one pass.
        if _HAS_JAX and getattr(self.config, "gradient_checkpointing", False):
            self._building = True
            try:
                seq = (
                    input_shape[1] if len(input_shape) > 1 else 8
                )
                seq = min(int(seq), int(self.config.context_length))
                dummy = ops.zeros((1, seq), dtype="int32")
                _ = self(dummy)
            finally:
                self._building = False

    def call(self, ids, segment_ids=None, return_loss_inputs=False):
        """Forward pass.

        Args:
            ids:         ``[batch, seq]`` integer token IDs.
            segment_ids: optional ``[batch, seq]`` integer document IDs for
                         packed multi-doc shards. When provided, attention is
                         masked so each document attends only to itself
                         (R2 in the 2026-05-11 dossier). When None, attention
                         falls back to plain causal (single-doc-per-sequence).
            return_loss_inputs: when True, returns the tuple
                         ``(hidden_states, lm_head_weight, output_mult)``
                         instead of full logits. Used by the chunked-CE
                         training path to avoid materialising ``[B, S, V]``.
                         The output_mult is the muP LM-head multiplier
                         (1.0 when muP is disabled). 2026-05-12: added
                         per senior reviewer pushback on full-logit OOM.
        """
        # ids: [batch, seq]
        s = ops.shape(ids)[1]
        x = self.embed(ids)
        cos = self._rope_cos.value[:s]
        sin = self._rope_sin.value[:s]
        # When segment_ids is provided we let the attention layer build the
        # combined (causal & same-segment) mask; otherwise pass a precomputed
        # causal mask as before.
        mask = None if segment_ids is not None else causal_mask(s, dtype=x.dtype)

        # Gradient checkpointing per block (when enabled + JAX backend).
        # 2026-05-12: required to fit 1B model at seq=8192 on a single H200.
        # Each block's forward runs twice (once at forward, once recomputed
        # during backward) but backward-stored activations drop ~4-8x,
        # which lets us fit larger contexts/batches.
        #
        # Skip during the variable-build warm-up pass (build() sets
        # self._building=True) so lazy-init Variables don't escape the
        # jax.checkpoint trace.
        use_ckpt = (
            bool(getattr(self.config, "gradient_checkpointing", False))
            and _HAS_JAX
            and not getattr(self, "_building", False)
        )
        for block in self.blocks:
            if use_ckpt:
                import jax  # local import; we just confirmed _HAS_JAX above
                # Closure captures cos/sin/mask/segment_ids as constants;
                # only x flows through the checkpointed boundary, so only
                # the per-block input is saved (not all intermediates).
                def _run(x_in, _block=block):
                    return _block(
                        x_in, cos, sin, mask=mask, segment_ids=segment_ids,
                    )
                x = jax.checkpoint(_run)(x)
            else:
                x = block(x, cos, sin, mask=mask, segment_ids=segment_ids)
        x = self.final_norm(x)

        # muP LM-head output multiplier (no-op when muP disabled).
        if (
            self.config.mup is not None
            and self.config.mup.apply_lm_head_output_mult
        ):
            output_mult = 1.0 / self.config.mup_width_multiplier()
        else:
            output_mult = 1.0

        if return_loss_inputs:
            # Chunked-CE path: hand the loss function the raw materials
            # (hidden states + LM-head weight + muP mult) instead of full
            # logits. The loss streams the vocab in chunks; we never
            # materialise [B, S, V] in either forward or backward.
            if self.config.tie_embeddings:
                lm_head_w = self.embed.embeddings  # [V, H]
            else:
                # untied: self.lm_head.kernel is [H, V]; transpose to [V, H].
                lm_head_w = ops.transpose(self.lm_head.kernel)
            return x, lm_head_w, ops.cast(output_mult, x.dtype)

        # Default path: apply the LM head, return full [B, S, V] logits.
        if self.config.tie_embeddings:
            logits = ops.matmul(x, ops.transpose(self.embed.embeddings))
        else:
            logits = self.lm_head(x)
        if output_mult != 1.0:
            logits = logits * output_mult
        return logits

    @classmethod
    def from_yaml(cls, path: str) -> "TransformerLM":
        return cls(ModelConfig.from_yaml(path))


def build_model(config: ModelConfig) -> TransformerLM:
    """Convenience: instantiate and build a model so weights exist immediately."""
    model = TransformerLM(config)
    # Trigger build via a dummy forward.
    dummy = ops.zeros((1, min(8, config.context_length)), dtype="int32")
    _ = model(dummy)
    return model
