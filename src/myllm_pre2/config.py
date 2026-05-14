"""Validated pre-2 config contracts.

These schemas validate planning configs before the TorchTitan implementation
exists. They deliberately do not parse the older pre-1 flat YAML files.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from myllm_pre2.guards import HETEROGENEOUS_TOPK_KD


class AttentionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["gqa"]
    num_heads: int = Field(gt=0)
    num_kv_heads: int = Field(gt=0)
    head_dim: int = Field(gt=0)

    @model_validator(mode="after")
    def _check_heads(self) -> "AttentionConfig":
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        return self


class PositionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["rope"]
    rope_base: int = Field(gt=0)
    rope_base_ablations: list[int] = Field(default_factory=list)


class EmbeddingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tied: bool = True


class TokenizerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidates: list[int] = Field(min_length=1)
    decision: str | None = None
    current_reference_vocab_size: int | None = Field(default=None, gt=0)

    def planning_vocab_size(self) -> int:
        """Return the vocab size used for parameter planning."""
        return self.current_reference_vocab_size or max(self.candidates)


class ContextConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    foundation_length: int = Field(gt=0)
    continuation_lengths: list[int] = Field(default_factory=list)


class DenseModelSection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    architecture: Literal["decoder_only_transformer"]
    parameter_target: int = Field(gt=0)
    layers: int = Field(gt=0)
    hidden_dim: int = Field(gt=0)
    ffn_dim: int = Field(gt=0)
    activation: Literal["swiglu"]
    norm: Literal["rmsnorm"]
    norm_eps: float = Field(gt=0)
    qk_norm: bool
    attention: AttentionConfig
    position: PositionConfig
    embeddings: EmbeddingConfig
    tokenizer: TokenizerConfig
    context: ContextConfig

    @model_validator(mode="after")
    def _check_shape(self) -> "DenseModelSection":
        expected_hidden = self.attention.num_heads * self.attention.head_dim
        if expected_hidden != self.hidden_dim:
            raise ValueError(
                f"num_heads * head_dim ({expected_hidden}) must equal hidden_dim "
                f"({self.hidden_dim})"
            )
        if self.ffn_dim % self.hidden_dim != 0:
            raise ValueError("ffn_dim should be an integer multiple of hidden_dim")
        return self


class ObjectiveConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    next_token_ce: bool = True
    z_loss_coef: float | None = Field(default=None, ge=0)
    z_loss_coef_ablations: list[float] = Field(default_factory=list)
    disabled_paths: list[str] = Field(default_factory=list)


class OptimizerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["adamw"]
    beta1: float = Field(gt=0, lt=1)
    beta2: float = Field(gt=0, lt=1)
    weight_decay: float = Field(ge=0)
    eps: float | None = Field(default=None, gt=0)
    grad_clip_global_norm: float = Field(gt=0)


class SchedulerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["wsd"]
    peak_lr: float | None = Field(default=None, gt=0)
    peak_lr_ablations: list[float] = Field(default_factory=list)
    lr_ablations: list[float] = Field(default_factory=list)
    warmup_steps_range: list[int] = Field(min_length=2, max_length=2)
    stable_fraction: float | None = Field(default=None, gt=0, lt=1)
    decay_fraction: float = Field(gt=0, lt=1)
    end_lr_ratio: float | None = Field(default=None, ge=0)


class BatchConfig(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    global_batch_tokens: int | None = Field(default=None, gt=0)
    sequence_length: int | None = Field(default=None, gt=0)


class TokenBudgetConfig(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    internal_minimum: int | None = Field(default=None, gt=0)
    release_target: int | None = Field(default=None, gt=0)
    stretch: int | None = Field(default=None, gt=0)
    research_ceiling: int | None = Field(default=None, gt=0)
    smoke: int | None = Field(default=None, gt=0)
    study_minimum: int | None = Field(default=None, gt=0)
    study_target: int | None = Field(default=None, gt=0)
    study_stretch: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _check_ordering(self) -> "TokenBudgetConfig":
        ordered_groups = [
            ("internal_minimum", "release_target", "stretch", "research_ceiling"),
            ("smoke", "study_minimum", "study_target", "study_stretch"),
        ]
        for group in ordered_groups:
            present = [(name, getattr(self, name)) for name in group if getattr(self, name) is not None]
            for (left_name, left), (right_name, right) in zip(present, present[1:], strict=False):
                if left is not None and right is not None and right < left:
                    raise ValueError(f"{right_name} must be >= {left_name}")
        return self


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dtype: Literal["bf16"]
    objective: ObjectiveConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    batch: BatchConfig | None = None
    token_budget: TokenBudgetConfig

    @model_validator(mode="after")
    def _check_learning_rates(self) -> "TrainingConfig":
        rates = []
        if self.scheduler.peak_lr is not None:
            rates.append(self.scheduler.peak_lr)
        rates.extend(self.scheduler.peak_lr_ablations)
        rates.extend(self.scheduler.lr_ablations)
        if not rates:
            raise ValueError("at least one peak LR or LR ablation is required")
        return self


class DensePre2Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: float | str
    status: Literal["planning"]
    name: str
    stack: Literal["torchtitan"]
    model: DenseModelSection
    training: TrainingConfig
    parallelism: dict[str, Any]
    precision_policy: dict[str, Any] | None = None
    gates: dict[str, list[str]] | None = None
    study_outputs: list[str] | None = None

    def parameter_count_estimate(self) -> int:
        """Estimate dense params using the same planning math as the ADR."""
        vocab = self.model.tokenizer.planning_vocab_size()
        hidden = self.model.hidden_dim
        layers = self.model.layers
        ffn = self.model.ffn_dim
        kv_dim = self.model.attention.num_kv_heads * self.model.attention.head_dim

        embedding = vocab * hidden
        if not self.model.embeddings.tied:
            embedding *= 2

        attention_per_layer = hidden * hidden + 2 * hidden * kv_dim + hidden * hidden
        ffn_per_layer = 3 * hidden * ffn
        qk_norm = 2 * self.model.attention.head_dim if self.model.qk_norm else 0
        norms = 2 * hidden
        return embedding + layers * (attention_per_layer + ffn_per_layer + qk_norm + norms)

    def parameter_target_delta(self) -> float:
        """Relative distance between estimate and config target."""
        return abs(self.parameter_count_estimate() - self.model.parameter_target) / (
            self.model.parameter_target
        )

    def training_steps_for_tokens(self, tokens: int) -> int | None:
        """Return optimizer steps for a token budget if global batch is known."""
        batch = self.training.batch
        if batch is None or batch.global_batch_tokens is None:
            return None
        return math.ceil(tokens / batch.global_batch_tokens)

    def dense_training_flops(self, tokens: int) -> int:
        """Approximate dense pretraining FLOPs with the standard 6*N*T rule."""
        return 6 * self.parameter_count_estimate() * tokens

    @model_validator(mode="after")
    def _check_disabled_kd(self) -> "DensePre2Config":
        disabled = set(self.training.objective.disabled_paths)
        if self.name.endswith("1.5b-base") and HETEROGENEOUS_TOPK_KD not in disabled:
            raise ValueError("mainline pre-2 config must disable hetero-tokenizer top-K KD")
        return self


class SourceBucketConfig(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    bucket: str
    target_share: float | None = Field(default=None, ge=0, le=1)
    target_tokens: int | None = Field(default=None, gt=0)
    source_ids: list[str] = Field(default_factory=list)
    gates: list[str] = Field(default_factory=list)
    allowed_range: list[float] | None = Field(default=None, min_length=2, max_length=2)
    share_range: list[float] | None = Field(default=None, min_length=2, max_length=2)

    @model_validator(mode="after")
    def _check_ranges(self) -> "SourceBucketConfig":
        for field_name in ("allowed_range", "share_range"):
            value = getattr(self, field_name)
            if value is not None and value[0] > value[1]:
                raise ValueError(f"{field_name} lower bound must be <= upper bound")
        if self.target_share is not None and self.allowed_range is not None:
            lo, hi = self.allowed_range
            if not lo <= self.target_share <= hi:
                raise ValueError("target_share must fall inside allowed_range")
        return self


class DataMixConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: float | str
    status: Literal["planning"]
    name: str
    target_tokens: int = Field(gt=0)
    synthetic_cap: float = Field(ge=0, le=0.10)
    source_buckets: list[SourceBucketConfig] = Field(min_length=1)
    required_metadata: list[str] | None = None
    entry_gates: list[str] | None = None
    exit_artifacts: list[str] | None = None
    anneal_rules: dict[str, Any] | None = None
    optional_context_continuation: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _check_target_shares(self) -> "DataMixConfig":
        shares = [bucket.target_share for bucket in self.source_buckets]
        if all(share is not None for share in shares):
            total = sum(float(share) for share in shares if share is not None)
            if abs(total - 1.0) > 1.0e-6:
                raise ValueError(f"target_share values must sum to 1.0, got {total:.6f}")
        return self


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not parse to a mapping")
    return data


def load_dense_config(path: str | Path) -> DensePre2Config:
    """Load a pre-2 dense model planning config."""
    data = _load_yaml(path)
    return DensePre2Config.model_validate(data)


def load_data_mix_config(path: str | Path) -> DataMixConfig:
    """Load a pre-2 data-mix planning config."""
    data = _load_yaml(path)
    return DataMixConfig.model_validate(data)
