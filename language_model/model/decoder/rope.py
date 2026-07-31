"""Rotary position embedding and its inverse gradient rotation."""

from __future__ import annotations

import numpy as np


def _rotate(values: np.ndarray, cosine: np.ndarray, sine: np.ndarray) -> np.ndarray:
    rotated = np.empty_like(values)
    rotated[..., 0::2] = values[..., 0::2] * cosine - values[..., 1::2] * sine
    rotated[..., 1::2] = values[..., 0::2] * sine + values[..., 1::2] * cosine
    return rotated


def apply_rope(
    q: np.ndarray, k: np.ndarray, theta: float = 10_000.0
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Rotate Q and K, returning the sine/cosine cache needed by backpropagation."""
    if q.shape != k.shape or q.ndim != 4:
        raise ValueError("q and k must have shape [B, heads, T, head_dim].")
    if q.shape[-1] % 2 or theta <= 0:
        raise ValueError("head_dim must be even and theta must be positive.")

    sequence_length, head_dim = q.shape[-2:]
    positions = np.arange(sequence_length, dtype=q.dtype)
    frequencies = theta ** (-np.arange(0, head_dim, 2, dtype=q.dtype) / head_dim)
    angles = positions[:, None] * frequencies
    cosine = np.cos(angles)[None, None]
    sine = np.sin(angles)[None, None]
    return _rotate(q, cosine, sine), _rotate(k, cosine, sine), (cosine, sine)


def rope_backward(
    q_gradient: np.ndarray, k_gradient: np.ndarray, cache: tuple[np.ndarray, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the transpose of the RoPE rotation to Q and K gradients."""
    cosine, sine = cache

    def inverse_rotate(gradient: np.ndarray) -> np.ndarray:
        values = np.empty_like(gradient)
        values[..., 0::2] = gradient[..., 0::2] * cosine + gradient[..., 1::2] * sine
        values[..., 1::2] = -gradient[..., 0::2] * sine + gradient[..., 1::2] * cosine
        return values

    return inverse_rotate(q_gradient), inverse_rotate(k_gradient)
