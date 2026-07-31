"""Vocabulary projection and its explicit backward pass."""

from __future__ import annotations

import numpy as np

from .layers import flatten, layer_norm, layer_norm_backward


def init_lm_head(
    hidden_size: int, vocabulary_size: int, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    return {
        "w_vocab": rng.normal(
            0.0, 1.0 / np.sqrt(hidden_size), (hidden_size, vocabulary_size)
        ).astype(np.float32),
        "b_vocab": np.zeros(vocabulary_size, dtype=np.float32),
    }


def lm_head(
    hidden_states: np.ndarray, parameters: dict[str, np.ndarray]
) -> tuple[np.ndarray, tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]]:
    normalized, norm_cache = layer_norm(hidden_states)
    return normalized @ parameters["w_vocab"] + parameters["b_vocab"], (normalized, norm_cache)


def lm_head_backward(
    gradient: np.ndarray,
    cache: tuple[np.ndarray, tuple[np.ndarray, np.ndarray]],
    parameters: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    normalized, norm_cache = cache
    gradients = {
        "w_vocab": flatten(normalized).T @ flatten(gradient),
        "b_vocab": gradient.sum(axis=(0, 1)),
    }
    return layer_norm_backward(gradient @ parameters["w_vocab"].T, norm_cache), gradients
