"""The tiny supervised classification loop shared by both vision scripts."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from .loss import loss


def _run(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = total_correct = total_examples = 0
    with torch.set_grad_enabled(training):
        for images, targets in loader:
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            value = loss(logits, targets)
            if training:
                value.backward()
                optimizer.step()
            total_loss += value.item() * len(targets)
            total_correct += (logits.argmax(dim=-1) == targets).sum().item()
            total_examples += len(targets)
    return total_loss / total_examples, total_correct / total_examples


def train_epoch(model: nn.Module, loader: DataLoader, device: torch.device, optimizer: torch.optim.Optimizer) -> tuple[float, float]:
    return _run(model, loader, device, optimizer)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    return _run(model, loader, device)
