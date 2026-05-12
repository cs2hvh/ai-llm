"""Validated model configuration loaded from YAML."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class MupConfig(BaseModel):
    """Maximal Update Parameterization (muP) settings.

    When attached to a ``ModelConfig``, the model uses muP scaling so that
    HPs (LR, init) tuned at small scale transfer zero-shot to larger widths.
    See ``docs/mup_design.md`` for the full recipe.

    When ``ModelConfig.mup`` is ``None``, the model runs in standard
    parameterization (SP) and behaves exactly as before muP was added —
    every change in ``layers.py`` is gated on the presence of this config.

    ``base_width`` is the hidden_dim at which HPs were tuned (the
    wind-tunnel width, typically 256-384). The width multiplier for this
    model is ``hidden_dim / base_width``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    base_width: int = Field(gt=0, le=4096)
    # Output multipliers active. Setting all three to True is "full muP";
    # setting all to False makes muP a no-op (useful for ablation).
    apply_attn_output_mult: bool = True
    apply_ffn_output_mult: bool = True
    apply_lm_head_output_mult: bool = True


class ModelConfig(BaseModel):
    """Architecture spec for a decoder-only transformer.

    Mirrors `configs/pilot_250m.yaml` and `configs/base_1b.yaml`. Used by the
    model builder, the trainer, and the HF/GGUF exporters as the single source
    of truth for shape parameters.
    """

    name: str
    arch: Literal["llama_decoder"] = "llama_decoder"

    layers: int = Field(gt=0)
    hidden_dim: int = Field(gt=0)
    ffn_dim: int = Field(gt=0)
    num_heads: int = Field(gt=0)
    num_kv_heads: int = Field(gt=0)
    head_dim: int = Field(gt=0)
    vocab_size: int = Field(gt=0)
    tie_embeddings: bool = True

    context_length: int = Field(gt=0)
    context_extension_target: int | None = None

    position: Literal["rope"] = "rope"
    rope_base: float = 10000.0

    norm: Literal["rmsnorm"] = "rmsnorm"
    norm_eps: float = 1.0e-5
    activation: Literal["swiglu"] = "swiglu"

    z_loss_coef: float = 1.0e-4
    qk_norm: bool = False

    init_std: float = 0.02
    scaled_init_for_residuals: bool = False

    # Activation checkpointing (jax.checkpoint per DecoderBlock).
    #
    # When True, the backward pass recomputes each block's forward activations
    # instead of storing them. Cuts backward-stored activation memory ~4-8×
    # at the cost of ~33% more compute (each block forward runs twice:
    # once at forward, once at backward).
    #
    # 2026-05-12 (1B benchmark bisection on 5xH200): without this, the 1B
    # model OOMed at compile time on seq=8192 (XLA requested 1.37 TB). With
    # this on, the 1B model at seq=8192 fits at mb=1 on a single H200 and
    # has measurable throughput. Default True for production safety; turn
    # off only for small models where the 33% recompute tax outweighs the
    # memory savings.
    gradient_checkpointing: bool = True

    # Optional muP / muTransfer parameterization. When set, the model
    # applies the muP scaling factors documented in `docs/mup_design.md`
    # to enable zero-shot HP transfer from a small proxy to this model.
    # When None, the model uses standard parameterization (no behavior change).
    mup: MupConfig | None = None

    def mup_width_multiplier(self) -> float:
        """Returns ``hidden_dim / mup.base_width``, or 1.0 if muP not enabled."""
        if self.mup is None:
            return 1.0
        return self.hidden_dim / self.mup.base_width

    @model_validator(mode="after")
    def _check_head_consistency(self) -> "ModelConfig":
        if self.num_heads * self.head_dim != self.hidden_dim:
            raise ValueError(
                f"num_heads * head_dim ({self.num_heads * self.head_dim}) "
                f"!= hidden_dim ({self.hidden_dim})"
            )
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by "
                f"num_kv_heads ({self.num_kv_heads}) for GQA"
            )
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ModelConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        # Strip non-architecture sections that may live alongside in the same YAML.
        arch_keys = set(cls.model_fields.keys())
        return cls(**{k: v for k, v in data.items() if k in arch_keys})

    def param_count_estimate(self) -> int:
        """Rough parameter count, ignoring norms and biases."""
        v, h, l, ff = self.vocab_size, self.hidden_dim, self.layers, self.ffn_dim
        kv = self.num_kv_heads * self.head_dim
        embed = v * h if self.tie_embeddings else 2 * v * h
        attn_per_layer = h * h + 2 * h * kv + h * h  # q, k, v (small via GQA), o
        ffn_per_layer = 3 * h * ff  # SwiGLU: gate, up, down
        return embed + l * (attn_per_layer + ffn_per_layer)
