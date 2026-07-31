"""One causal Transformer block with explicit NumPy backpropagation."""

from __future__ import annotations

import numpy as np

from .layers import Activation, flatten, layer_norm, layer_norm_backward, softmax
from .rope import apply_rope, rope_backward


def init_block(
    hidden_size: int, heads: int, activation: str, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    if hidden_size <= 0 or heads <= 0 or hidden_size % heads or (hidden_size // heads) % 2:
        raise ValueError("hidden_size must be positive, divisible by heads, and have an even head dimension.")
    scale = 1.0 / np.sqrt(hidden_size)
    mlp_size = 4 * hidden_size
    projection_size = Activation.projection_size(mlp_size, activation)

    def weights(shape: tuple[int, int]) -> np.ndarray:
        return rng.normal(0.0, scale, shape).astype(np.float32)

    return {
        "wq": weights((hidden_size, hidden_size)),
        "wk": weights((hidden_size, hidden_size)),
        "wv": weights((hidden_size, hidden_size)),
        "wo": weights((hidden_size, hidden_size)),
        "w1": weights((hidden_size, projection_size)),
        "w2": weights((mlp_size, hidden_size)),
    }


def attention_block(
    values: np.ndarray, parameters: dict[str, np.ndarray], heads: int, activation: str
) -> tuple[np.ndarray, dict[str, object]]:
    if values.ndim != 3:
        raise ValueError("values must have shape [B, T, D].")
    batch_size, sequence_length, hidden_size = values.shape
    if heads <= 0 or hidden_size % heads or (hidden_size // heads) % 2:
        raise ValueError("hidden size must be divisible by heads with an even head dimension.")
    head_dim = hidden_size // heads

    def split_heads(projection: np.ndarray) -> np.ndarray:
        return projection.reshape(batch_size, sequence_length, heads, head_dim).transpose(0, 2, 1, 3)

    normalized_1, norm_1_cache = layer_norm(values)
    q = split_heads(normalized_1 @ parameters["wq"])
    k = split_heads(normalized_1 @ parameters["wk"])
    v = split_heads(normalized_1 @ parameters["wv"])
    q, k, rope_cache = apply_rope(q, k)

    causal = np.tril(np.ones((sequence_length, sequence_length), dtype=bool))
    scores = q @ k.swapaxes(-1, -2) / np.sqrt(head_dim)
    weights = softmax(np.where(causal[None, None], scores, -np.inf))
    context = weights @ v
    merged_context = context.transpose(0, 2, 1, 3).reshape(batch_size, sequence_length, hidden_size)
    after_attention = values + merged_context @ parameters["wo"]

    normalized_2, norm_2_cache = layer_norm(after_attention)
    mlp_pre_activation = normalized_2 @ parameters["w1"]
    mlp_hidden = Activation.forward(mlp_pre_activation, activation)
    output = after_attention + mlp_hidden @ parameters["w2"]
    return output, {
        "normalized_1": normalized_1,
        "norm_1": norm_1_cache,
        "q": q,
        "k": k,
        "v": v,
        "rope": rope_cache,
        "weights": weights,
        "causal": causal,
        "merged_context": merged_context,
        "normalized_2": normalized_2,
        "norm_2": norm_2_cache,
        "mlp_pre_activation": mlp_pre_activation,
        "mlp_hidden": mlp_hidden,
    }


def attention_block_backward(
    output_gradient: np.ndarray,
    cache: dict[str, object],
    parameters: dict[str, np.ndarray],
    heads: int,
    activation: str,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    normalized_1 = cache["normalized_1"]
    q = cache["q"]
    k = cache["k"]
    v = cache["v"]
    weights = cache["weights"]
    causal = cache["causal"]
    merged_context = cache["merged_context"]
    normalized_2 = cache["normalized_2"]
    mlp_pre_activation = cache["mlp_pre_activation"]
    mlp_hidden = cache["mlp_hidden"]
    norm_1_cache = cache["norm_1"]
    norm_2_cache = cache["norm_2"]
    rope_cache = cache["rope"]

    after_attention_gradient = output_gradient.copy()
    mlp_pre_activation_gradient = Activation.backward(
        output_gradient @ parameters["w2"].T, mlp_pre_activation, activation
    )
    gradients = {
        "w2": flatten(mlp_hidden).T @ flatten(output_gradient),
        "w1": flatten(normalized_2).T @ flatten(mlp_pre_activation_gradient),
    }
    normalized_2_gradient = mlp_pre_activation_gradient @ parameters["w1"].T
    after_attention_gradient += layer_norm_backward(normalized_2_gradient, norm_2_cache)

    input_gradient = after_attention_gradient.copy()
    gradients["wo"] = flatten(merged_context).T @ flatten(after_attention_gradient)
    merged_context_gradient = after_attention_gradient @ parameters["wo"].T
    batch_size, sequence_length, hidden_size = merged_context_gradient.shape
    head_dim = hidden_size // heads
    context_gradient = merged_context_gradient.reshape(
        batch_size, sequence_length, heads, head_dim
    ).transpose(0, 2, 1, 3)
    weights_gradient = context_gradient @ v.swapaxes(-1, -2)
    v_gradient = weights.swapaxes(-1, -2) @ context_gradient
    scores_gradient = weights * (
        weights_gradient - (weights_gradient * weights).sum(axis=-1, keepdims=True)
    )
    scores_gradient *= causal[None, None]
    scale = 1.0 / np.sqrt(head_dim)
    q_gradient = scores_gradient @ k * scale
    k_gradient = scores_gradient.swapaxes(-1, -2) @ q * scale
    q_gradient, k_gradient = rope_backward(q_gradient, k_gradient, rope_cache)

    q_gradient = q_gradient.transpose(0, 2, 1, 3).reshape(batch_size, sequence_length, hidden_size)
    k_gradient = k_gradient.transpose(0, 2, 1, 3).reshape(batch_size, sequence_length, hidden_size)
    v_gradient = v_gradient.transpose(0, 2, 1, 3).reshape(batch_size, sequence_length, hidden_size)
    gradients["wq"] = flatten(normalized_1).T @ flatten(q_gradient)
    gradients["wk"] = flatten(normalized_1).T @ flatten(k_gradient)
    gradients["wv"] = flatten(normalized_1).T @ flatten(v_gradient)
    normalized_1_gradient = (
        q_gradient @ parameters["wq"].T
        + k_gradient @ parameters["wk"].T
        + v_gradient @ parameters["wv"].T
    )
    input_gradient += layer_norm_backward(normalized_1_gradient, norm_1_cache)
    return input_gradient, gradients
