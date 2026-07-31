"""Loss functions for image classification."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    """Compute ordinary multiclass cross-entropy with explicit shape checks."""
    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, classes].")
    if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
        raise ValueError("targets must have shape [batch] matching logits.")
    if targets.dtype != torch.long:
        raise ValueError("targets must use torch.long class indices.")
    return F.cross_entropy(logits, targets)
