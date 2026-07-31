"""Feed-forward and residual layers shared by the ViT encoder blocks."""

from __future__ import annotations

from torch import Tensor, nn

from .attention import MultiHeadSelfAttention


class FeedForward(nn.Module):
    """The fixed GELU MLP used after every attention layer."""

    def __init__(self, hidden_size: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        if hidden_size <= 0 or mlp_ratio <= 0:
            raise ValueError("hidden_size and mlp_ratio must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the interval [0, 1).")

        intermediate_size = max(1, int(hidden_size * mlp_ratio))
        self.input_projection = nn.Linear(hidden_size, intermediate_size)
        self.activation = nn.GELU()
        self.output_projection = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.input_projection(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.output_projection(hidden_states)
        return self.dropout(hidden_states)


class TransformerBlock(nn.Module):
    """A pre-normalized ViT block with attention and a GELU feed-forward path."""

    def __init__(
        self,
        hidden_size: int,
        heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        rope_theta: float = 10_000.0,
    ) -> None:
        super().__init__()
        self.pre_attention_norm = nn.LayerNorm(hidden_size)
        self.attention = MultiHeadSelfAttention(
            hidden_size=hidden_size,
            heads=heads,
            dropout=attention_dropout,
            rope_theta=rope_theta,
        )
        self.pre_mlp_norm = nn.LayerNorm(hidden_size)
        self.mlp = FeedForward(hidden_size, mlp_ratio, dropout)

    def forward(
        self,
        hidden_states: Tensor,
        grid_size: int | tuple[int, int],
    ) -> Tensor:
        hidden_states = hidden_states + self.attention(
            self.pre_attention_norm(hidden_states), grid_size
        )
        return hidden_states + self.mlp(self.pre_mlp_norm(hidden_states))
