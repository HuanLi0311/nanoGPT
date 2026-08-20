"""ImageFolder loading and model-specific image preprocessing."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


class ProcessorTransform:
    def __init__(self, processor) -> None:
        self.processor = processor

    def __call__(self, image):
        return self.processor(images=image, return_tensors="pt")["pixel_values"][0]


def resolve_device(name: str) -> torch.device:
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loaders(
    train_dir: str | Path,
    validation_dir: str | Path,
    transform,
    batch_size: int,
    workers: int,
    seed: int,
) -> tuple[DataLoader, DataLoader, list[str]]:
    train = ImageFolder(str(train_dir), transform=transform)
    validation = ImageFolder(str(validation_dir), transform=transform)
    generator = torch.Generator().manual_seed(seed)
    options = {"batch_size": batch_size, "num_workers": workers, "pin_memory": torch.cuda.is_available()}
    return (
        DataLoader(train, shuffle=True, generator=generator, **options),
        DataLoader(validation, shuffle=False, **options),
        list(train.classes),
    )
