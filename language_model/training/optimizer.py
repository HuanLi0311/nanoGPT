"""AdamW and Muon updates for the parameters selected by train.yaml."""

import numpy as np


class Optimizer:
    def __init__(self, parameters: dict[str, np.ndarray], learning_rate: float) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.step_count = 0
        self.first_moment = {name: np.zeros_like(value) for name, value in parameters.items()}
        self.second_moment = {name: np.zeros_like(value) for name, value in parameters.items()}
        self.momentum = {name: np.zeros_like(value) for name, value in parameters.items()}

    def _check_gradients(self, gradients: dict[str, np.ndarray]) -> None:
        if set(gradients) != set(self.parameters):
            raise ValueError("Gradients must match the parameter names.")

    def _adamw_update(
        self,
        name: str,
        gradient: np.ndarray,
        beta1: float,
        beta2: float,
        epsilon: float,
        weight_decay: float,
    ) -> None:
        first = self.first_moment[name] = (
            beta1 * self.first_moment[name] + (1.0 - beta1) * gradient
        )
        second = self.second_moment[name] = (
            beta2 * self.second_moment[name] + (1.0 - beta2) * gradient**2
        )
        parameter = self.parameters[name]
        parameter *= 1.0 - self.learning_rate * weight_decay
        parameter -= self.learning_rate * (first / (1.0 - beta1**self.step_count)) / (
            np.sqrt(second / (1.0 - beta2**self.step_count)) + epsilon
        )

    def adamw(
        self,
        gradients: dict[str, np.ndarray],
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1 or epsilon <= 0 or weight_decay < 0:
            raise ValueError("Invalid AdamW hyperparameters.")
        self._check_gradients(gradients)
        self.step_count += 1
        for name, gradient in gradients.items():
            self._adamw_update(name, gradient, beta1, beta2, epsilon, weight_decay)

    def muon(self, gradients: dict[str, np.ndarray], momentum: float = 0.95, weight_decay: float = 0.01) -> None:
        """Use Muon on matrices and AdamW on the vector biases it cannot orthogonalize."""
        if not 0 <= momentum < 1 or weight_decay < 0:
            raise ValueError("Invalid Muon hyperparameters.")
        self._check_gradients(gradients)
        self.step_count += 1
        for name, parameter in self.parameters.items():
            if parameter.ndim != 2:
                self._adamw_update(name, gradients[name], 0.9, 0.999, 1e-8, weight_decay)
                continue
            update = self.momentum[name] = momentum * self.momentum[name] + gradients[name]
            left, singular_values, right = np.linalg.svd(update, full_matrices=False)
            tolerance = np.finfo(update.dtype).eps * max(update.shape) * singular_values[0]
            parameter *= 1.0 - self.learning_rate * weight_decay
            parameter -= self.learning_rate * ((left * (singular_values > tolerance)) @ right)

    def step(self, gradients: dict[str, np.ndarray], name: str) -> None:
        if name == "AdamW":
            self.adamw(gradients)
        elif name == "Muon":
            self.muon(gradients)
        else:
            raise ValueError(f"Unknown optimizer: {name}")


def clip_gradients(gradients: dict[str, np.ndarray], maximum_norm: float) -> float:
    if maximum_norm <= 0:
        raise ValueError("maximum_norm must be positive.")
    total_norm = float(np.sqrt(sum(np.sum(gradient**2) for gradient in gradients.values())))
    if total_norm > maximum_norm:
        for gradient in gradients.values():
            gradient *= maximum_norm / total_norm
    return total_norm
