"""Muon optimizer."""

from collections.abc import Iterable

import torch
from torch import Tensor


@torch.no_grad()

def muon(parameter, state, lr):

    for parameter in parameter:
        if parameter.grad is None:
            continue
        gradient = parameter.grad
        parameter_state = state.setdefault(parameter, {})
        parameter_state["step"] = parameter_state.get("step", 0) + 1

        if parameter.ndim != 2:
            first = parameter_state.setdefault("first", torch.zeros_like(parameter))
            second = parameter_state.setdefault("second", torch.zeros_like(parameter))
            first.mul_(0.9).add_(gradient, alpha=0.1)
            second.mul_(0.999).addcmul_(gradient, gradient, value=0.001)
            first_hat = first / (1.0 - 0.9 ** parameter_state["step"])
            second_hat = second / (1.0 - 0.999 ** parameter_state["step"])
            parameter.mul_(1.0 - lr * 0.01)
            parameter.addcdiv_(first_hat, second_hat.sqrt().add_(1e-8), value=-lr)
            continue
        
        update = parameter_state.setdefault("momentum", torch.zeros_like(parameter))
        update.mul_(0.95).add_(gradient)
        left, singular_values, right = torch.linalg.svd(update, full_matrices=False)
        tolerance = torch.finfo(update.dtype).eps * max(update.shape) * singular_values[0]
        orthogonal = (left * (singular_values > tolerance)) @ right
        parameter.mul_(1.0 - lr * 0.01)
        parameter.add_(orthogonal, alpha=-lr)


class Optimizer:

    def __init__(self, parameters: Iterable[Tensor], learning_rate: float) -> None:
        self.parameters = list(parameters)
        self.state = {}
        self.learning_rate = learning_rate

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.grad = None

    def step(self) -> None:
        muon(self.parameters, self.state, self.learning_rate)
