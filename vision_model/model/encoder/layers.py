"""Transformer layers for the from-scratch ViT."""

from __future__ import annotations

from torch import Tensor, nn

from .attention import MultiHeadSelfAttention


class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        intermediate_size = int(hidden_size * ratio)
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.layers(hidden_states)


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, heads: int, ratio: float, dropout: float) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.attn = MultiHeadSelfAttention(hidden_size, heads, dropout)
        self.mlp_norm = nn.LayerNorm(hidden_size)
        self.mlp = FeedForward(hidden_size, ratio, dropout)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = hidden_states + self.attn(self.attn_norm(hidden_states))
        return hidden_states + self.mlp(self.mlp_norm(hidden_states))
