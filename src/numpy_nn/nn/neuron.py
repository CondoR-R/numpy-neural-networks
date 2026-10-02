import numpy as np


class MPNeuron:
    def __init__(self, weights: np.typing.NDArray[np.float32], threshold: np.float32) -> None:
        self.weights   = weights
        self.threshold = threshold


    def __call__(self, x: np.ndarray) -> np.typing.NDArray[np.uint8]:
        z = x @ self.weights
        return (z >= self.threshold).astype(np.uint8)
