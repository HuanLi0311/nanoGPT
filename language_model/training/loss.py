"""Stable vocabulary cross-entropy for next-token language-model training."""

import numpy as np


def cross_entropy(logits: np.ndarray, target_ids: np.ndarray) -> tuple[float, np.ndarray]:
    """Return mean loss and d_loss/d_logits for logits [B, T, V]."""
    if logits.ndim != 3 or target_ids.shape != logits.shape[:2]:
        raise ValueError("Expected logits [B, T, V] and target IDs [B, T].")
    if np.any(target_ids < 0) or np.any(target_ids >= logits.shape[-1]):
        raise ValueError("A target token ID is outside the logits vocabulary.")

    shifted = logits - logits.max(axis=-1, keepdims=True)
    log_probabilities = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    batch_ids = np.arange(logits.shape[0])[:, None]
    position_ids = np.arange(logits.shape[1])[None, :]
    loss = float(-log_probabilities[batch_ids, position_ids, target_ids].mean())

    gradient = np.exp(log_probabilities)
    gradient[batch_ids, position_ids, target_ids] -= 1.0
    gradient /= target_ids.size
    return loss, gradient
