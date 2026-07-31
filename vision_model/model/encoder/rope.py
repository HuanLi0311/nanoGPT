"""Standard two-dimensional rotary position embeddings for patch tokens."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def _as_grid_size(grid_size: int | tuple[int, int]) -> tuple[int, int]:
    """Normalize a square or rectangular patch grid to ``(height, width)``."""
    if isinstance(grid_size, int):
        grid_size = (grid_size, grid_size)
    if len(grid_size) != 2 or min(grid_size) <= 0:
        raise ValueError("grid_size must contain two positive dimensions.")
    return int(grid_size[0]), int(grid_size[1])


def _validate_head_dim(head_dim: int) -> None:
    if head_dim <= 0 or head_dim % 4:
        raise ValueError("head_dim must be positive and divisible by 4 for 2D RoPE.")


def build_2d_rope_cache(
    grid_size: int | tuple[int, int],
    head_dim: int,
    theta: float = 10_000.0,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor]:
    """Build cosine and sine tables for a row-major patch grid.

    Half of each attention head is assigned to row coordinates and the other
    half to column coordinates.  Each axis uses the usual interleaved rotary
    pairs, so a head dimension divisible by four is required.
    """
    height, width = _as_grid_size(grid_size)
    _validate_head_dim(head_dim)
    if theta <= 0:
        raise ValueError("theta must be positive.")
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise ValueError("dtype must be floating point.")

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


def apply_2d_rope(
    query: Tensor,
    key: Tensor,
    grid_size: int | tuple[int, int],
    theta: float = 10_000.0,
) -> tuple[Tensor, Tensor]:
    """Rotate query and key tensors with the same patch-grid coordinates.

    ``query`` and ``key`` have shape ``[batch, heads, patches, head_dim]``.
    Patches are interpreted in row-major ``grid_size`` order.
    """
    if query.shape != key.shape or query.ndim != 4:
        raise ValueError("query and key must have shape [B, heads, tokens, head_dim].")
    if not query.is_floating_point() or not key.is_floating_point():
        raise ValueError("query and key must use floating-point tensors.")
    if query.device != key.device or query.dtype != key.dtype:
        raise ValueError("query and key must use the same device and dtype.")
    height, width = _as_grid_size(grid_size)
    _validate_head_dim(query.shape[-1])
    expected_tokens = height * width
    if query.shape[-2] != expected_tokens:
        raise ValueError(
            f"patch count must be {expected_tokens} for grid {height}x{width}."
        )

    cosine, sine = build_2d_rope_cache(
        (height, width),
        query.shape[-1],
        theta,
        device=query.device,
        dtype=query.dtype,
    )
    return _rotate_pairs(query, cosine, sine), _rotate_pairs(key, cosine, sine)


class RotaryEmbedding2D(nn.Module):
    """Reusable module wrapper around :func:`apply_2d_rope`."""

    def __init__(self, head_dim: int, theta: float = 10_000.0) -> None:
        super().__init__()
        _validate_head_dim(head_dim)
        if theta <= 0:
            raise ValueError("theta must be positive.")
        self.head_dim = head_dim
        self.theta = theta

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        grid_size: int | tuple[int, int],
    ) -> tuple[Tensor, Tensor]:
        if query.shape[-1] != self.head_dim:
            raise ValueError("query head dimension does not match the RoPE configuration.")
        return apply_2d_rope(
            query,
            key,
            grid_size,
            theta=self.theta,
        )
