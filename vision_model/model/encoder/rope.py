"""Standard two-dimensional rotary position embeddings for patch tokens."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def _as_grid_size(grid_size: int | tuple[int, int]) -> tuple[int, int]:
    """Return square or rectangular patch-grid dimensions."""
    if isinstance(grid_size, int):
        return grid_size, grid_size
    return int(grid_size[0]), int(grid_size[1])


def build_2d_rope_cache(
    grid_size: int | tuple[int, int],
    head_dim: int,
    theta: float = 10_000.0,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor]:

    height, width = _as_grid_size(grid_size)

    axis_dim = head_dim // 2
    exponent = torch.arange(0, axis_dim, 2, device=device, dtype=torch.float32) / axis_dim
    inverse_frequency = torch.exp(-math.log(theta) * exponent)

    rows = torch.arange(height, device=device, dtype=torch.float32)
    columns = torch.arange(width, device=device, dtype=torch.float32)
    row_grid, column_grid = torch.meshgrid(rows, columns, indexing="ij")
    row_positions = row_grid.reshape(-1, 1)
    column_positions = column_grid.reshape(-1, 1)
    angles = torch.cat(
        (row_positions * inverse_frequency, column_positions * inverse_frequency), dim=-1
    )
    cosine = angles.cos().to(dtype=dtype).view(1, 1, height * width, axis_dim)
    sine = angles.sin().to(dtype=dtype).view(1, 1, height * width, axis_dim)
    return cosine, sine


def _rotate_pairs(values: Tensor, cosine: Tensor, sine: Tensor) -> Tensor:
    """Apply interleaved two-dimensional rotary rotations to the last axis."""
    rotated = torch.empty_like(values)
    even = values[..., 0::2]
    odd = values[..., 1::2]
    rotated[..., 0::2] = even * cosine - odd * sine
    rotated[..., 1::2] = even * sine + odd * cosine
    return rotated


def rope_2d(
    query: Tensor,
    key: Tensor,
    grid_size: int | tuple[int, int],
    theta: float = 10_000.0,
) -> tuple[Tensor, Tensor]:
    """Apply the same two-dimensional rotary positions to query and key."""

    cosine, sine = build_2d_rope_cache(
        grid_size,
        query.shape[-1],
        theta,
        device=query.device,
        dtype=query.dtype,
    )
    return _rotate_pairs(query, cosine, sine), _rotate_pairs(key, cosine, sine)


class RotaryEmbedding2D(nn.Module):
    """Reusable module wrapper around :func:`2d_rope`."""

    def __init__(self, head_dim: int, theta: float = 10_000.0) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        grid_size: int | tuple[int, int],
    ) -> tuple[Tensor, Tensor]:
        q, k = rope_2d(q, k, grid_size, theta=self.theta)
        return q, k
