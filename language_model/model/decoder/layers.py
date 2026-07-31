"""Small NumPy operations shared by the Transformer passes."""

from __future__ import annotations

import numpy as np


def flatten(values: np.ndarray) -> np.ndarray:
    return values.reshape(-1, values.shape[-1])


def layer_norm(values: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    centered = values - values.mean(axis=-1, keepdims=True)
    inverse_std = 1.0 / np.sqrt((centered**2).mean(axis=-1, keepdims=True) + 1e-5)
    normalized = centered * inverse_std
    return normalized, (normalized, inverse_std)


def layer_norm_backward(gradient: np.ndarray, cache: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    normalized, inverse_std = cache
    dimension = gradient.shape[-1]
    return inverse_std / dimension * (
        dimension * gradient
        - gradient.sum(axis=-1, keepdims=True)
        - normalized * (gradient * normalized).sum(axis=-1, keepdims=True)
    )


def gelu(values: np.ndarray) -> np.ndarray:
    return 0.5 * values * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (values + 0.044715 * values**3)))


def gelu_backward(gradient: np.ndarray, values: np.ndarray) -> np.ndarray:
    coefficient = np.sqrt(2.0 / np.pi)
    inner = coefficient * (values + 0.044715 * values**3)
    tanh_inner = np.tanh(inner)
    derivative = 0.5 * (1.0 + tanh_inner) + 0.5 * values * (1.0 - tanh_inner**2) * coefficient * (
        1.0 + 0.134145 * values**2
    )
    return gradient * derivative


class Activation:
    """The activation options exposed by train.yaml."""

    names = {"ReLU", "GeLU", "SiLU", "SwiGLU"}

    @staticmethod
    def ReLU(values: np.ndarray) -> np.ndarray:
        return np.maximum(values, 0.0)

    @staticmethod
    def GeLU(values: np.ndarray) -> np.ndarray:
        return gelu(values)

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        dtype = values.dtype if np.issubdtype(values.dtype, np.floating) else np.float32
        result = np.empty_like(values, dtype=dtype)
        positive = values >= 0
        result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
        exponentials = np.exp(values[~positive])
        result[~positive] = exponentials / (1.0 + exponentials)
        return result

    @staticmethod
    def SiLU(values: np.ndarray) -> np.ndarray:
        return values * Activation._sigmoid(values)

    @staticmethod
    def SwiGLU(values: np.ndarray) -> np.ndarray:
        if values.shape[-1] % 2:
            raise ValueError("SwiGLU requires an even-sized last dimension.")
        gate, linear = np.split(values, 2, axis=-1)
        return Activation.SiLU(gate) * linear

    @classmethod
    def projection_size(cls, hidden_size: int, name: str) -> int:
        if name not in cls.names:
            raise ValueError(f"Unknown activation: {name}")
        return 2 * hidden_size if name == "SwiGLU" else hidden_size

    @classmethod
    def forward(cls, values: np.ndarray, name: str) -> np.ndarray:
        if name not in cls.names:
            raise ValueError(f"Unknown activation: {name}")
        return getattr(cls, name)(values)

    @classmethod
    def backward(cls, gradient: np.ndarray, values: np.ndarray, name: str) -> np.ndarray:
        if name == "ReLU":
            return gradient * (values > 0)
        if name == "GeLU":
            return gelu_backward(gradient, values)
        if name == "SiLU":
            sigmoid = cls._sigmoid(values)
            return gradient * (sigmoid + values * sigmoid * (1.0 - sigmoid))
        if name == "SwiGLU":
            gate, linear = np.split(values, 2, axis=-1)
            sigmoid = cls._sigmoid(gate)
            gate_gradient = gradient * linear * (sigmoid + gate * sigmoid * (1.0 - sigmoid))
            return np.concatenate((gate_gradient, gradient * gate * sigmoid), axis=-1)
        raise ValueError(f"Unknown activation: {name}")


def softmax(values: np.ndarray) -> np.ndarray:
    exponentials = np.exp(values - values.max(axis=-1, keepdims=True))
    return exponentials / exponentials.sum(axis=-1, keepdims=True)
