"""Feed-forward and residual layers shared by the ViT encoder blocks."""

from __future__ import annotations

from torch import Tensor, nn

from .attention import MultiHeadSelfAttention


class FeedForward(nn.Module):
    """The fixed GELU MLP used after every attention layer."""

    def __init__(self, hidden_size: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        intermediate_size = max(1, int(hidden_size * mlp_ratio))
        self.input_up_proj = nn.Linear(hidden_size, intermediate_size)
        self.activation = nn.GELU()
        self.up_proj = nn.Linear(intermediate_size, hidden_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.activation(self.up_proj(hidden_states))
        hidden_states = self.dropout(self.down_proj(hidden_states))
        return hidden_states



class EncoderBlock(nn.Module):
    """A pre-normalized ViT block with attention and a GELU feed-forward path."""

    def __init__(
        self,
        hidden_size: int,
        heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        theta: float = 10_000.0,
    ) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.attn = MultiHeadSelfAttention(
            hidden_size=hidden_size,
            heads=heads,
            dropout=attention_dropout,
            theta=theta,
        )
        self.mlp_norm = nn.LayerNorm(hidden_size)
        self.mlp = FeedForward(hidden_size, mlp_ratio, dropout)

    def forward(
        self,
        hidden_states: Tensor,
        grid_size: int | tuple[int, int],
    ) -> Tensor:
        hidden_states = hidden_states + self.attn(self.norm(hidden_states), grid_size)
        hidden_states = hidden_states + self.mlp(self.norm(hidden_states))
        return hidden_states
