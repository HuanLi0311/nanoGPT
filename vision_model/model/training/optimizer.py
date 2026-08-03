"""The single optimizer used by the first linear-probe training path."""

from __future__ import annotations

from torch import nn, optim


def _parameter_groups(module: nn.Module, weight_decay: float) -> list[dict[str, object]]:
    """Keep decay off one-dimensional parameters and biases, as in standard ViT training."""
    decay = []
    no_decay = []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    if not decay and not no_decay:
        raise ValueError("The optimizer received no trainable parameters.")
    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


def build_optimizer(
    module: nn.Module,
    learning_rate: float,
    weight_decay: float,
) -> optim.Optimizer:
    """Build the fixed AdamW optimizer for the trainable probe parameters."""
    if learning_rate <= 0 or weight_decay < 0:
        raise ValueError("learning_rate must be positive and weight_decay cannot be negative.")
    return optim.AdamW(
        _parameter_groups(module, weight_decay),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
