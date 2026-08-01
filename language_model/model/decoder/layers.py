"""Feed-forward layer used by each decoder block."""

from __future__ import annotations

import math

from .attention import CausalSelfAttention
from torch import Tensor, nn
from torch.nn import functional as F


class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        intermediate_size = 4 * hidden_size
        self.hidden_size = hidden_size
        self.up_proj = nn.Linear(hidden_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

        scale = 1.0 / math.sqrt(hidden_size)
        for layer in (self.up_proj, self.down_proj):
            nn.init.normal_(layer.weight, mean=0.0, std=scale)
            nn.init.zeros_(layer.bias)

    def forward(self, hidden_states: Tensor) -> Tensor:
        intermediate = self.up_proj(hidden_states)
        intermediate = F.silu(intermediate)
        return self.dropout(self.down_proj(intermediate))

class DecoderBlock(nn.Module):
    def __init__(self, hidden_size: int, heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.attn = CausalSelfAttention(hidden_size, heads, dropout)
        self.mlp_norm = nn.LayerNorm(hidden_size)
        self.mlp = FeedForward(hidden_size, dropout)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = hidden_states + self.attn(self.attn_norm(hidden_states))
        hidden_states = hidden_states + self.mlp(self.mlp_norm(hidden_states))

        return hidden_states
