"""Classification loss."""

from torch import Tensor
from torch.nn import functional as F


def loss(logits: Tensor, targets: Tensor) -> Tensor:
    return F.cross_entropy(logits, targets)
