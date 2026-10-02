from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def relu(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.where(x > 0, x, 0)


def relu_grad(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return (x > 0).astype(np.uint8)


def sigmoid(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return 1 / (1 + np.exp(-x))


def sigmoid_grad(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return sigmoid(x) * (1 - sigmoid(x))


def tanh(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def tanh_grad(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return 1 - tanh(x) ** 2

