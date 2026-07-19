"""Global Perceiver resampling for variable-sized primitive sets."""

from __future__ import annotations

import torch
from torch import nn


class _FeedForward(nn.Module):
    def __init__(self, dim: int, dropout: float, expansion: int = 4) -> None:
        super().__init__()
        hidden_dim = expansion * dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class _PerceiverLayer(nn.Module):
    """One pre-norm cross-attention, self-attention, and FFN block."""

    def __init__(self, dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.cross_query_norm = nn.LayerNorm(dim)
        self.cross_context_norm = nn.LayerNorm(dim)
        self.cross_attention = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.self_norm = nn.LayerNorm(dim)
        self.self_attention = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = _FeedForward(dim, dropout)

    def forward(
        self,
        latents: torch.Tensor,
        primitive_tokens: torch.Tensor,
        primitive_mask: torch.Tensor,
    ) -> torch.Tensor:
        cross_output, _ = self.cross_attention(
            query=self.cross_query_norm(latents),
            key=self.cross_context_norm(primitive_tokens),
            value=self.cross_context_norm(primitive_tokens),
            key_padding_mask=~primitive_mask,
            need_weights=False,
        )
        latents = latents + cross_output

        normalized = self.self_norm(latents)
        self_output, _ = self.self_attention(
            query=normalized,
            key=normalized,
            value=normalized,
            need_weights=False,
        )
        latents = latents + self_output
        return latents + self.ffn(self.ffn_norm(latents))


class PerceiverResampler(nn.Module):
    """Compress each primitive set into a fixed number of learned latent tokens.

    Args:
        dim: Primitive feature width.
        num_latents: Number of output tokens per sample.
        num_layers: Number of Perceiver blocks.
        num_heads: Attention head count.
        dropout: Attention and feed-forward dropout.

    Inputs use ``True`` in ``primitive_mask`` for real primitives. Every sample
    must contain at least one real primitive.
    """

    def __init__(
        self,
        dim: int,
        num_latents: int = 32,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_latents <= 0:
            raise ValueError(f"num_latents must be positive, got {num_latents}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if num_heads <= 0 or dim % num_heads != 0:
            raise ValueError(
                f"num_heads must be positive and divide dim; got dim={dim}, "
                f"num_heads={num_heads}"
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.dim = dim
        self.num_latents = num_latents
        self.queries = nn.Parameter(torch.empty(num_latents, dim))
        self.layers = nn.ModuleList(
            [_PerceiverLayer(dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.output_norm = nn.LayerNorm(dim)
        nn.init.normal_(self.queries, std=dim**-0.5)

    def forward(
        self,
        primitive_tokens: torch.Tensor,
        primitive_mask: torch.Tensor,
    ) -> torch.Tensor:
        if primitive_tokens.ndim != 3:
            raise ValueError(
                "primitive_tokens must have shape [B, N, D], "
                f"got {tuple(primitive_tokens.shape)}"
            )
        batch_size, primitive_count, dim = primitive_tokens.shape
        if dim != self.dim:
            raise ValueError(f"expected primitive dim {self.dim}, got {dim}")
        if primitive_mask.shape != (batch_size, primitive_count):
            raise ValueError(
                "primitive_mask must match the first two primitive token dimensions; "
                f"got {tuple(primitive_mask.shape)}"
            )
        if primitive_mask.dtype != torch.bool:
            raise TypeError(
                f"primitive_mask must be a BoolTensor, got {primitive_mask.dtype}"
            )
        if batch_size == 0:
            return primitive_tokens.new_empty((0, self.num_latents, self.dim))
        if not torch.all(primitive_mask.any(dim=-1)):
            raise ValueError("every resampled sample must contain at least one primitive")

        latents = self.queries.unsqueeze(0).expand(batch_size, -1, -1)

        for layer in self.layers:
            latents = layer(latents, primitive_tokens, primitive_mask)
        return self.output_norm(latents)


__all__ = ["PerceiverResampler"]
