"""Multi-head causal self-attention."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .rope import rope


####################################################################################
#                           Multi-Head Self-Attention                              #
####################################################################################

class MultiHeadSelfAttention(nn.Module):
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

    @staticmethod
    def split_heads(values: Tensor, heads: int) -> Tensor:
        batch_size, token_count, hidden_size = values.shape
        head_dim = hidden_size // heads
        return values.view(batch_size, token_count, heads, head_dim).transpose(1, 2)

    @staticmethod
    def merge_heads(values: Tensor) -> Tensor:
        batch_size, _, token_count, _ = values.shape
        return values.transpose(1, 2).contiguous().view(batch_size, token_count, -1)


    def forward(
        self, hidden_states: Tensor, past_key_value: tuple[Tensor, Tensor] | None = None,
        use_cache: bool = False,
    ) -> Tensor | tuple[Tensor, tuple[Tensor, Tensor]]:
        q, k, v = self.qkv(hidden_states).chunk(3, dim=-1)
        q = self.split_heads(q, self.heads)
        k = self.split_heads(k, self.heads)
        v = self.split_heads(v, self.heads)
        past_length = 0 if past_key_value is None else past_key_value[0].shape[-2]
        q, k = rope(q, k, past_length)
        if past_key_value is not None:
            k, v = torch.cat((past_key_value[0], k), -2), torch.cat((past_key_value[1], v), -2)
            positions = torch.arange(k.shape[-2], device=k.device)
            mask = positions <= past_length + torch.arange(q.shape[-2], device=q.device)[:, None]
            output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        output = self.output_proj(self.merge_heads(output))
        output = self.dropout(output)
        return (output, (k, v)) if use_cache else output
