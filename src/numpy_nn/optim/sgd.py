"""
Стохастический градиентный спуск.

Простейший оптимизатор, обновляющий каждый параметр в направлении,
противоположном его градиенту:

    p = p - lr * grad

Служит отправной точкой для сравнения с более сложными оптимизаторами
(Momentum, Adam и другие) и работает как базовый алгоритм обучения.
"""
from __future__ import annotations

from numpy_nn.nn.core import Parameter

from .base import Optimizer


class SGD(Optimizer):
    """
    Стохастический градиентный спуск (Stochastic Gradient Descent).

    Обновляет параметры по формуле ``p.data -= lr * p.grad``. Параметры
    с ``grad is None`` пропускаются — это нормальная ситуация, когда
    часть модели не участвовала в последнем forward.

    Название «стохастический» — историческое: спуск работает на
    mini-batch'ах, поэтому градиент на каждой итерации — это оценка
    истинного градиента по всей выборке, а не точное значение.

    Attributes:
        params: Список :class:`Parameter`, унаследованный от базового
            класса.
        lr: Learning rate.

    Example:
        >>> from numpy_nn.nn import Linear
        >>> layer = Linear(4, 3)
        >>> optimizer = SGD(layer.parameters(), lr=0.01)
        >>> optimizer.zero_grad()
        >>> # ... forward, backward ...
        >>> optimizer.step()
    """

    def __init__(self, params: list[Parameter], lr: float) -> None:
        """
        Инициализирует оптимизатор SGD.

        Args:
            params: Параметры для обновления.
            lr: Скорость обучения.
        """
        super().__init__(params, lr)

    def step(self) -> None:
        """
        Делает один шаг SGD.

        Для каждого параметра с непустым градиентом выполняет
        ``p.data -= lr * p.grad``. Параметры с ``grad is None``
        пропускаются — они не участвовали в последнем backward.
        """
        for param in self.params:
            if param.grad is None:
                continue

            param.data -= self.lr * param.grad
