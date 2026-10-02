import numpy as np
from numpy.typing import NDArray


class MPNeuron:
    def __init__(self, weights: NDArray[np.float32], threshold: float) -> None:
        self.weights = weights
        self.threshold = threshold

    def __call__(self, x: NDArray[np.float32]) -> NDArray[np.uint8]:
        z = x @ self.weights
        return (z >= self.threshold).astype(np.uint8)
