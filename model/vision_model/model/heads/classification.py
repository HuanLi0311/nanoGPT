"""The single classification head used by both vision paths."""

from __future__ import annotations

from torch import Tensor, nn


class ClassificationHead(nn.Module):
    def __init__(self, hidden_size: int, classes: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, classes)

    def forward(self, features: Tensor) -> Tensor:
        return self.proj(features)
