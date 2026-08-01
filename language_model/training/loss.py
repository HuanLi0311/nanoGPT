"""Cross-entropy loss for next-token training."""

from torch import Tensor
from torch.nn import functional as F


def cross_entropy(logits: Tensor, target_ids: Tensor) -> Tensor:
    loss = F.cross_entropy(logits.flatten(0, -2), target_ids.flatten())
    return loss
