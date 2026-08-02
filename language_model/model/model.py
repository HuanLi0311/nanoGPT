"""Decoder-only Transformer assembled from the model submodules."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .decoder.layers import DecoderBlock
from .decoder.lm_head import LMHead
from .tokenizer import embedding


class Transformer(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        max_sequence_length: int,
        hidden_size: int,
        heads: int,
        blocks: int,
        dropout: float = 0.0,
        seed: int = 7,
    ) -> None:
        super().__init__()

        torch.manual_seed(seed)
        self.vocabulary_size = vocabulary_size
        self.max_sequence_length = max_sequence_length
        self.hidden_size = hidden_size
        self.heads = heads
        self.block_count = blocks
        self.token_embedding = nn.Parameter(torch.empty(vocabulary_size, hidden_size))
        self.blocks = nn.ModuleList(
            [DecoderBlock(hidden_size, heads, dropout) for _ in range(blocks)]
        )
        self.lm_head = LMHead(hidden_size, vocabulary_size)
        nn.init.normal_(self.token_embedding, mean=0.0, std=0.02)

    def forward(self, token_ids: Tensor) -> Tensor:

        hidden_states = embedding(self.token_embedding, token_ids)
        for block in self.blocks:
            hidden_states = block(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits
