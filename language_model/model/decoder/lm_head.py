"""Final normalization and vocabulary up_proj."""

from __future__ import annotations

import math
from torch import Tensor, nn


class LMHead(nn.Module):
    def __init__(self, hidden_size: int, vocabulary_size: int) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.norm = nn.LayerNorm(hidden_size)
        self.proj = nn.Linear(hidden_size, vocabulary_size)
        nn.init.normal_(self.proj.weight, mean=0.0, std=1.0 / math.sqrt(hidden_size))
        nn.init.zeros_(self.proj.bias)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.norm(hidden_states)
        logits = self.proj(hidden_states)
        return logits
