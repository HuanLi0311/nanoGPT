"""Rotary position embedding."""

from __future__ import annotations

import torch
from torch import Tensor


def rope(
    query: Tensor, key: Tensor, theta: float = 10_000.0
) -> tuple[Tensor, Tensor]:
    token_count, head_dim = query.shape[-2:]
    positions = torch.arange(token_count, device=query.device, dtype=query.dtype)
    frequencies = theta ** (
        -torch.arange(0, head_dim, 2, device=query.device, dtype=query.dtype) / head_dim
    )
    angles = positions[:, None] * frequencies
    cosine = angles.cos()[None, None]
    sine = angles.sin()[None, None]

    def rotate(values: Tensor) -> Tensor:
        even, odd = values[..., 0::2], values[..., 1::2]
        return torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), -1).flatten(-2)

    return rotate(query), rotate(key)
