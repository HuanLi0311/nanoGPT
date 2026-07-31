"""Token embedding lookup and its scatter-add gradient."""

import numpy as np


def embedding(table: np.ndarray, token_ids: np.ndarray) -> np.ndarray:
    return table[token_ids]


def embedding_backward(
    table: np.ndarray, token_ids: np.ndarray, gradient: np.ndarray
) -> np.ndarray:
    table_gradient = np.zeros_like(table)
    np.add.at(table_gradient, token_ids, gradient)
    return table_gradient
