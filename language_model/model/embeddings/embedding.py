"""Token embedding lookup."""

from torch import Tensor


def embedding(token_embedding: Tensor, token_ids: Tensor) -> Tensor:
    hidden_states = token_embedding[token_ids]
    return hidden_states
