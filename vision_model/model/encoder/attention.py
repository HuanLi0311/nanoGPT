"""Multi-head self-attention used by the teaching ViT encoder."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .rope import RotaryEmbedding2D


class MultiHeadSelfAttention(nn.Module):
    """Explicit Q/K/V attention with a shared two-dimensional RoPE module."""

    def __init__(
        self,
        hidden_size: int,
        heads: int,
        dropout: float = 0.0,
        rope_theta: float = 10_000.0,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or heads <= 0 or hidden_size % heads:
            raise ValueError("hidden_size must be positive and divisible by heads.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the interval [0, 1).")

        self.hidden_size = hidden_size
        self.heads = heads
        self.head_dim = hidden_size // heads
        self.query_projection = nn.Linear(hidden_size, hidden_size)
        self.key_projection = nn.Linear(hidden_size, hidden_size)
        self.value_projection = nn.Linear(hidden_size, hidden_size)
        self.output_projection = nn.Linear(hidden_size, hidden_size)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding2D(self.head_dim, rope_theta)

    def _split_heads(self, values: Tensor) -> Tensor:
        batch_size, token_count, _ = values.shape
        return values.view(batch_size, token_count, self.heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, values: Tensor) -> Tensor:
        batch_size, _, token_count, _ = values.shape
        return values.transpose(1, 2).contiguous().view(batch_size, token_count, self.hidden_size)

    def forward(
        self,
        hidden_states: Tensor,
        grid_size: int | tuple[int, int],
    ) -> Tensor:
        """Return attended states with shape ``[batch, tokens, hidden_size]``."""
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError("hidden_states must have shape [B, tokens, hidden_size].")

        query = self._split_heads(self.query_projection(hidden_states))
        key = self._split_heads(self.key_projection(hidden_states))
        value = self._split_heads(self.value_projection(hidden_states))
        query, key = self.rope(query, key, grid_size)

        scores = torch.matmul(query, key.transpose(-2, -1)) * (self.head_dim**-0.5)
        weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(dtype=query.dtype)
        weights = self.attention_dropout(weights)
        context = torch.matmul(weights, value)
        merged_context = self._merge_heads(context)
        return self.output_dropout(self.output_projection(merged_context))
