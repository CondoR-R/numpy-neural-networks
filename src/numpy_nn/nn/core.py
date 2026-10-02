from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class Parameter:
    def __init__(self, data: NDArray[np.float64]) -> None:
        self.data: NDArray[np.float64] = np.array(data, dtype=np.float64, copy=True)
        self.grad: NDArray[np.float64] | None = None

    def zero_grad(self) -> None:
        self.grad = None

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape


class Module:
    def __init__(self) -> None:
        pass

    def __call__(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.forward(x)

    def forward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        raise NotImplementedError(
            'forward должен быть реализован в наследнике, иначе ничего не работает'
        )

    def backward(self, dout: NDArray[np.float64]) -> NDArray[np.float64]:
        raise NotImplementedError(
            'backward должен быть реализован в наследнике, иначе ничего не работает'
        )

    def parameters(self) -> list[Parameter]:
        return []

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()
