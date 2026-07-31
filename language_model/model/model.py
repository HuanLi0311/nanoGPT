"""A small decoder-only Transformer: one model definition for training and inference."""

from __future__ import annotations

import numpy as np

from .decoder.attention import attention_block, attention_block_backward, init_block
from .decoder.lm_head import init_lm_head, lm_head, lm_head_backward
from .embeddings.embedding import embedding, embedding_backward


class Transformer:
    def __init__(
        self,
        vocabulary_size: int,
        max_sequence_length: int,
        hidden_size: int,
        heads: int,
        blocks: int,
        seed: int = 7,
        activation: str = "GeLU",
    ) -> None:
        if min(vocabulary_size, max_sequence_length, hidden_size, heads, blocks) <= 0:
            raise ValueError("All model dimensions must be positive.")
        if hidden_size % heads or (hidden_size // heads) % 2:
            raise ValueError("hidden_size must be divisible by heads with an even head dimension.")

        rng = np.random.default_rng(seed)
        self.vocabulary_size = vocabulary_size
        self.max_sequence_length = max_sequence_length
        self.hidden_size = hidden_size
        self.heads = heads
        self.block_count = blocks
        self.activation = activation
        self.token_embedding = rng.normal(
            0.0, 0.02, (vocabulary_size, hidden_size)
        ).astype(np.float32)
        self.blocks = [init_block(hidden_size, heads, activation, rng) for _ in range(blocks)]
        self.lm_head = init_lm_head(hidden_size, vocabulary_size, rng)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.size for parameter in self.parameters().values())

    @property
    def configuration(self) -> dict[str, int | str]:
        return {
            "vocabulary_size": self.vocabulary_size,
            "max_sequence_length": self.max_sequence_length,
            "hidden_size": self.hidden_size,
            "heads": self.heads,
            "blocks": self.block_count,
            "activation": self.activation,
        }

    def forward(self, token_ids: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
        token_ids = np.asarray(token_ids)
        if not np.issubdtype(token_ids.dtype, np.integer):
            raise ValueError("token_ids must be integers.")
        if token_ids.ndim == 1:
            token_ids = token_ids[None]
        if token_ids.ndim != 2 or not 0 < token_ids.shape[1] <= self.max_sequence_length:
            raise ValueError("token_ids must have shape [B, T] with 0 < T <= max_sequence_length.")
        token_ids = token_ids.astype(np.int64, copy=False)
        if np.any(token_ids < 0) or np.any(token_ids >= self.vocabulary_size):
            raise ValueError("A token ID is outside the vocabulary.")

        values = embedding(self.token_embedding, token_ids)
        block_caches = []
        for parameters in self.blocks:
            values, cache = attention_block(values, parameters, self.heads, self.activation)
            block_caches.append(cache)
        logits, head_cache = lm_head(values, self.lm_head)
        return logits, {"input_ids": token_ids, "blocks": block_caches, "head": head_cache}

    def backward(self, cache: dict[str, object], logits_gradient: np.ndarray) -> dict[str, np.ndarray]:
        input_ids = cache["input_ids"]
        if logits_gradient.shape != (input_ids.shape[0], input_ids.shape[1], self.vocabulary_size):
            raise ValueError("logits_gradient must match the logits returned by forward.")

        values_gradient, head_gradients = lm_head_backward(logits_gradient, cache["head"], self.lm_head)
        gradients = {f"lm_{name}": value for name, value in head_gradients.items()}
        for index in reversed(range(self.block_count)):
            values_gradient, block_gradients = attention_block_backward(
                values_gradient, cache["blocks"][index], self.blocks[index], self.heads, self.activation
            )
            gradients.update({f"block_{index}_{name}": value for name, value in block_gradients.items()})
        gradients["token_embedding"] = embedding_backward(
            self.token_embedding, input_ids, values_gradient
        )
        return gradients

    def parameters(self) -> dict[str, np.ndarray]:
        parameters = {"token_embedding": self.token_embedding}
        for index, block in enumerate(self.blocks):
            parameters.update({f"block_{index}_{name}": value for name, value in block.items()})
        parameters.update({f"lm_{name}": value for name, value in self.lm_head.items()})
        return parameters
