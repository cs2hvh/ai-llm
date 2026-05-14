"""Minimal pre-2 PyTorch dense model.

This is intentionally a small PyTorch module, not a TorchTitan trainer. It
exists to make the pre-2 config executable enough for shape, loss, and guard
tests before distributed training is introduced.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from myllm_pre2.config import DensePre2Config
from myllm_pre2.guards import reject_topk_kd_inputs


@dataclass(frozen=True)
class Pre2ForwardOutput:
    logits: Tensor
    loss: Tensor | None = None


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


def _rotate_half_interleaved(x: Tensor) -> Tensor:
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply RoPE to ``[batch, heads, seq, head_dim]`` tensors."""
    return (x * cos) + (_rotate_half_interleaved(x) * sin)


def rope_cache(seq_len: int, head_dim: int, base: int, *, device: torch.device) -> tuple[Tensor, Tensor]:
    if head_dim % 2 != 0:
        raise ValueError("RoPE head_dim must be even")
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(positions, inv_freq)
    emb = torch.repeat_interleave(freqs, repeats=2, dim=-1)
    return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]


def _weighted_mean(values: Tensor, weights: Tensor) -> Tensor:
    denom = weights.sum()
    numerator = (values * weights).sum()
    return torch.where(denom > 0, numerator / denom.clamp_min(1.0), numerator * 0.0)


class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg: DensePre2Config) -> None:
        super().__init__()
        model = cfg.model
        attn = model.attention
        self.hidden_dim = model.hidden_dim
        self.num_heads = attn.num_heads
        self.num_kv_heads = attn.num_kv_heads
        self.head_dim = attn.head_dim
        self.rope_base = model.position.rope_base
        self.q_norm = RMSNorm(self.head_dim, model.norm_eps) if model.qk_norm else None
        self.k_norm = RMSNorm(self.head_dim, model.norm_eps) if model.qk_norm else None

        kv_dim = self.num_kv_heads * self.head_dim
        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_dim, kv_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_dim, kv_dim, bias=False)
        self.o_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        cos, sin = rope_cache(seq_len, self.head_dim, self.rope_base, device=x.device)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if self.num_heads != self.num_kv_heads:
            repeats = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, self.hidden_dim)
        return self.o_proj(y)


class SwiGLU(nn.Module):
    def __init__(self, hidden_dim: int, ffn_dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, hidden_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderBlock(nn.Module):
    def __init__(self, cfg: DensePre2Config) -> None:
        super().__init__()
        hidden = cfg.model.hidden_dim
        eps = cfg.model.norm_eps
        self.attn_norm = RMSNorm(hidden, eps)
        self.attn = GroupedQueryAttention(cfg)
        self.ffn_norm = RMSNorm(hidden, eps)
        self.ffn = SwiGLU(hidden, cfg.model.ffn_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.attn_norm(x))
        return x + self.ffn(self.ffn_norm(x))


class DenseTransformerLM(nn.Module):
    def __init__(self, cfg: DensePre2Config) -> None:
        super().__init__()
        self.cfg = cfg
        model = cfg.model
        vocab_size = model.tokenizer.planning_vocab_size()
        self.token_embedding = nn.Embedding(vocab_size, model.hidden_dim)
        self.layers = nn.ModuleList(DecoderBlock(cfg) for _ in range(model.layers))
        self.final_norm = RMSNorm(model.hidden_dim, model.norm_eps)
        self.lm_head = nn.Linear(model.hidden_dim, vocab_size, bias=False)
        if model.embeddings.tied:
            self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: Tensor,
        *,
        labels: Tensor | None = None,
        loss_mask: Tensor | None = None,
        teacher_topk_logits: Tensor | None = None,
        teacher_topk_indices: Tensor | None = None,
    ) -> Pre2ForwardOutput:
        """Run the dense LM.

        ``labels`` are expected to be already shifted and aligned with
        ``input_ids``. This matches the packed-corpus contract where each input
        position predicts the corresponding label position and ``loss_mask``
        suppresses document-boundary or padded positions.
        """
        reject_topk_kd_inputs(
            teacher_topk_logits=teacher_topk_logits,
            teacher_topk_indices=teacher_topk_indices,
        )
        if loss_mask is not None and labels is None:
            raise ValueError("loss_mask requires labels")

        x = self.token_embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        logits = self.lm_head(self.final_norm(x))

        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError(
                    "labels must be shifted/aligned to input_ids and have the same shape"
                )
            if loss_mask is not None and loss_mask.shape != labels.shape:
                raise ValueError("loss_mask must have the same shape as labels")

            valid = labels.ne(-100).to(dtype=logits.dtype)
            if loss_mask is not None:
                valid = valid * loss_mask.to(device=logits.device, dtype=logits.dtype)

            per_token_loss = F.cross_entropy(
                logits.contiguous().view(-1, logits.size(-1)),
                labels.contiguous().view(-1),
                ignore_index=-100,
                reduction="none",
            ).view_as(labels)
            loss = _weighted_mean(per_token_loss, valid)
            z_loss_coef = self.cfg.training.objective.z_loss_coef
            if z_loss_coef:
                log_z = torch.logsumexp(logits, dim=-1)
                loss = loss + z_loss_coef * _weighted_mean(log_z.pow(2), valid)

        return Pre2ForwardOutput(logits=logits, loss=loss)

    def parameter_count(self) -> int:
        return sum(param.numel() for param in self.parameters())


def build_dense_lm(cfg: DensePre2Config) -> DenseTransformerLM:
    return DenseTransformerLM(cfg)
