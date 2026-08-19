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

    def forward(
        self, token_ids: Tensor, past_key_values: list[tuple[Tensor, Tensor]] | None = None,
        use_cache: bool = False,
    ) -> Tensor | tuple[Tensor, list[tuple[Tensor, Tensor]]]:
        hidden_states = embedding(self.token_embedding, token_ids)
        if use_cache:
            past_key_values = past_key_values or [None] * len(self.blocks)
            present_key_values = []
            for block, past_key_value in zip(self.blocks, past_key_values):
                hidden_states, present_key_value = block(hidden_states, past_key_value, True)
                present_key_values.append(present_key_value)
        else:
            for block in self.blocks:
                hidden_states = block(hidden_states)
        logits = self.lm_head(hidden_states)
        return (logits, present_key_values) if use_cache else logits
