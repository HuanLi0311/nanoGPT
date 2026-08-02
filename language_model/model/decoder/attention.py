"""Multi-head causal self-attention."""

from __future__ import annotations

import math

from torch import Tensor, nn
from torch.nn import functional as F

from .rope import rope


def split_heads(values: Tensor, heads: int) -> Tensor:
    batch_size, token_count, hidden_size = values.shape
    head_dim = hidden_size // heads
    return values.view(batch_size, token_count, heads, head_dim).transpose(1, 2)


def merge_heads(values: Tensor) -> Tensor:
    batch_size, _, token_count, _ = values.shape
    return values.transpose(1, 2).contiguous().view(batch_size, token_count, -1)


####################################################################################
#                           Multi-Head Self-Attention                              #
####################################################################################

class CausalSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.heads = heads
        self.head_dim = hidden_size // heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

        scale = 1.0 / math.sqrt(hidden_size)
        for layer in (self.qkv, self.output_proj):
            nn.init.normal_(layer.weight, mean=0.0, std=scale)
            nn.init.zeros_(layer.bias)

    def forward(self, hidden_states: Tensor) -> Tensor:
        q, k, v = self.qkv(hidden_states).chunk(3, dim=-1)
        q = split_heads(q, self.heads)
        k = split_heads(k, self.heads)
        v = split_heads(v, self.heads)
        q, k = rope(q, k)
        output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        output = self.output_proj(merge_heads(output))
        output = self.dropout(output)
        return output
