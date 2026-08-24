"""Cross-entropy loss for next-token training."""

from torch import Tensor
from torch.nn import functional as F


def pretrain_loss(logits: Tensor, target_ids: Tensor) -> Tensor:
    log_p = F.log_softmax(logits, dim=-1)
    target_log_p = log_p.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    return -target_log_p.mean()


def sft_loss(logits: Tensor, target_ids: Tensor) -> Tensor:
    supervised = target_ids.ne(-100)
    if not supervised.any():
        return logits.sum() * 0.0
    log_p = logits.softmax(dim=-1).log()
    safe_target_ids = target_ids.masked_fill(~supervised, 0)
    target_log_p = log_p.gather(-1, safe_target_ids.unsqueeze(-1)).squeeze(-1)
    return -target_log_p[supervised].mean()