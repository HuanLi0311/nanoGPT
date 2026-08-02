"""Multi-head self-attention for the from-scratch ViT."""

from __future__ import annotations

from torch import Tensor, nn
from torch.nn import functional as F


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = hidden_size // heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = dropout
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: Tensor) -> Tensor:
        batch_size, token_count, hidden_size = hidden_states.shape
        q, k, v = self.qkv(hidden_states).reshape(
            batch_size, token_count, 3, self.heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        output = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )
        output = output.transpose(1, 2).reshape(batch_size, token_count, hidden_size)
        return self.output_dropout(self.proj(output))
