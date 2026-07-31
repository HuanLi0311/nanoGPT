"""Configuration loading and reproducibility helpers."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_settings(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping and check the sections used by the trainer."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        settings = yaml.safe_load(stream)
    if not isinstance(settings, dict):
        raise ValueError("The YAML root must be a mapping.")

    required_sections = {"data", "model", "training", "checkpoint"}
    missing = sorted(required_sections.difference(settings))
    if missing:
        raise ValueError(f"Configuration is missing sections: {', '.join(missing)}")
    return settings


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable experiments."""
    if seed < 0:
        raise ValueError("seed must be non-negative.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(name: str = "auto") -> torch.device:
    """Resolve ``auto`` to CUDA when available and otherwise use the CPU."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    return device
