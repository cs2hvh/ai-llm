"""Full TransformerLM assembly: token embed → N decoder blocks → norm → lm_head."""
from __future__ import annotations

import keras
from keras import ops

from myllm.model.config import ModelConfig
from myllm.model.layers import DecoderBlock, RMSNorm, precompute_rope_cache


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

    def call(self, ids, segment_ids=None):
        """Forward pass.

        Args:
            ids:         ``[batch, seq]`` integer token IDs.
            segment_ids: optional ``[batch, seq]`` integer document IDs for
                         packed multi-doc shards. When provided, attention is
                         masked so each document attends only to itself
                         (R2 in the 2026-05-11 dossier). When None, attention
                         falls back to plain causal (single-doc-per-sequence).
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
        for block in self.blocks:
            x = block(x, cos, sin, mask=mask, segment_ids=segment_ids)
        x = self.final_norm(x)
        if self.config.tie_embeddings:
            # Tie lm_head weight to the embedding matrix.
            logits = ops.matmul(x, ops.transpose(self.embed.embeddings))
        else:
            logits = self.lm_head(x)
        # muP LM-head output multiplier (no-op when muP disabled).
        if (
            self.config.mup is not None
            and self.config.mup.apply_lm_head_output_mult
        ):
            logits = logits * (1.0 / self.config.mup_width_multiplier())
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
