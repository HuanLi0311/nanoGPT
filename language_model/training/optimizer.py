"""PyTorch optimizers selected by train.yaml."""

from collections.abc import Iterable

import torch
from torch import Tensor


class _Muon(torch.optim.Optimizer):
    def __init__(self, parameters: Iterable[Tensor], learning_rate: float) -> None:
        super().__init__(
            parameters,
            {"lr": learning_rate, "momentum": 0.95, "weight_decay": 0.01},
        )

    @torch.no_grad()
    def step(self) -> None:
        for group in self.param_groups:
            learning_rate = group["lr"]
            momentum_rate = group["momentum"]
            weight_decay = group["weight_decay"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                state = self.state[parameter]
                state["step"] = state.get("step", 0) + 1
                if parameter.ndim != 2:
                    first = state.setdefault("first", torch.zeros_like(parameter))
                    second = state.setdefault("second", torch.zeros_like(parameter))
                    first.mul_(0.9).add_(gradient, alpha=0.1)
                    second.mul_(0.999).addcmul_(gradient, gradient, value=0.001)
                    first_hat = first / (1.0 - 0.9 ** state["step"])
                    second_hat = second / (1.0 - 0.999 ** state["step"])
                    parameter.mul_(1.0 - learning_rate * weight_decay)
                    parameter.addcdiv_(
                        first_hat, second_hat.sqrt().add_(1e-8), value=-learning_rate
                    )
                    continue
                update = state.setdefault("momentum_buffer", torch.zeros_like(parameter))
                update.mul_(momentum_rate).add_(gradient)
                left, singular_values, right = torch.linalg.svd(update, full_matrices=False)
                tolerance = torch.finfo(update.dtype).eps * max(update.shape) * singular_values[0]
                orthogonal = (left * (singular_values > tolerance)) @ right
                parameter.mul_(1.0 - learning_rate * weight_decay)
                parameter.add_(orthogonal, alpha=-learning_rate)


class Optimizer:
    def __init__(self, parameters: Iterable[Tensor], learning_rate: float) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        self.parameters = list(parameters)
        self.learning_rate = learning_rate
        self._name = None
        self._optimizer = None

    def zero_grad(self) -> None:
        if self._optimizer is not None:
            self._optimizer.zero_grad(set_to_none=True)

    def step(self, name: str) -> None:
        if name != self._name:
            if name == "AdamW":
                self._optimizer = torch.optim.AdamW(
                    self.parameters, lr=self.learning_rate, weight_decay=0.01
                )
            elif name == "Muon":
                self._optimizer = _Muon(self.parameters, self.learning_rate)
            else:
                raise ValueError(f"Unknown optimizer: {name}")
            self._name = name
        self._optimizer.step()
