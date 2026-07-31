"""A minimal linear-probe head for frozen visual features."""

from __future__ import annotations

from torch import Tensor, nn


class LinearProbe(nn.Module):
    """Pool patch features and classify them with one trainable linear layer."""

    def __init__(self, hidden_size: int, num_classes: int) -> None:
        super().__init__()
        if hidden_size <= 0 or num_classes <= 1:
            raise ValueError("hidden_size must be positive and num_classes must exceed one.")
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, features: Tensor) -> Tensor:
        """Return logits from ``[B, tokens, hidden]`` or ``[B, hidden]`` features."""
        if features.ndim == 3:
            features = features.mean(dim=1)
        if features.ndim != 2 or features.shape[-1] != self.hidden_size:
            raise ValueError("features must have shape [B, hidden] or [B, tokens, hidden].")
        return self.classifier(features)
