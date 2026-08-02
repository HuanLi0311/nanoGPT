"""Multi-head self-attention used by the teaching ViT encoder."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .rope import RotaryEmbedding2D


def split_heads(self, values: Tensor) -> Tensor:
    batch_size, token_count, _ = values.shape
    return values.view(batch_size, token_count, self.heads, self.head_dim).transpose(1, 2)

def merge_heads(self, values: Tensor) -> Tensor:
    batch_size, _, token_count, _ = values.shape
    return values.transpose(1, 2).contiguous().view(batch_size, token_count, self.hidden_size)


class MultiHeadSelfAttention(nn.Module):
    """Explicit Q/K/V attention with a shared two-dimensional RoPE module."""

    def __init__(self, hidden_size: int, heads: int, dropout: float = 0.0, theta: float = 10_000.0) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.heads = heads
        self.head_dim = hidden_size // heads
        self.qkv = nn.Linear(hidden_size, 3*hidden_size)
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding2D(self.head_dim, theta)

    def forward(
        self,
        hidden_states: Tensor,
        grid_size: int | tuple[int, int],
    ) -> Tensor:
        q, k, v = self.qkv(hidden_states).chunk(3, dim=-1)
        q, k = self.rope(q, k, grid_size)
        output = F.scaled_dot_product_attention(q, k, v, is_casual=True)
        output = self.output_proj(self.merge_heads(output))
        output = self.dropout(output)
        return output
