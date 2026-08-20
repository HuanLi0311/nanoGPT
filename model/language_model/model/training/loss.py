"""Cross-entropy loss for next-token training."""

from torch import Tensor
from torch.nn import functional as F

def pretrain_loss(logits: Tensor, target_ids: Tensor) -> Tensor:
       loss = F.cross_entropy(logits.flatten(0, -2), target_ids.flatten())
       return loss


def sft_loss(logits: Tensor, target_ids: Tensor) -> Tensor:
    if not target_ids.ne(-100).any():
          return logits.sum() * 0.0
    loss = F.cross_entropy(
          logits.flatten(0, -2), 
          target_ids.flatten(),
          ignore_index=-100,
          )
    return loss
