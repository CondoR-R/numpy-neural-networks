"""
Оптимизатор Momentum.

Идея: вместо шага строго против текущего градиента накапливать
экспоненциально сглаженную «скорость» и двигаться по ней. Это гасит
колебания в оврагах и ускоряет движение вдоль пологих направлений.

Формула (в стиле PyTorch):

    v = beta * v + grad
    p = p - lr * v

В отличие от классической записи с множителем ``(1 - beta)``
перед градиентом, здесь градиент добавляется без коэффициента —
это эквивалентная форма, используемая в PyTorch.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from numpy_nn.nn.core import Parameter

from .base import Optimizer


class Momentum(Optimizer):
    """
    Стохастический градиентный спуск с моментом.

    Хранит для каждого параметра буфер скорости той же формы. На
    каждом шаге буфер сглаживается коэффициентом ``beta`` и к нему
    добавляется текущий градиент. Параметр обновляется по буферу,
    а не по мгновенному градиенту.

    Параметры с ``grad is None`` полностью пропускаются: их буфер
    не обновляется и значение не меняется. Это соответствует
    поведению PyTorch.

    Attributes:
        params: Список :class:`Parameter` из базового класса.
        lr: Learning rate.
        beta: Коэффициент сглаживания буфера скорости. Обычно ``0.9``.
            Значение должно быть в диапазоне ``[0, 1)``.
        v: Список буферов скорости, по одному на каждый параметр.
            Форма каждого совпадает с формой соответствующего
            ``param.data``.

    Example:
        >>> from numpy_nn.nn import Linear
        >>> layer = Linear(4, 3)
        >>> optimizer = Momentum(layer.parameters(), lr=0.01, beta=0.9)
        >>> optimizer.zero_grad()
        >>> # ... forward, backward ...
        >>> optimizer.step()
    """

    def __init__(
        self,
        params: list[Parameter],
        lr: float,
        beta: float = 0.9,
    ) -> None:
        """
        Инициализирует оптимизатор Momentum.

        Args:
            params: Параметры для обновления.
            lr: Скорость обучения.
            beta: Коэффициент сглаживания буфера скорости.

        Raises:
            ValueError: Если ``beta`` вне диапазона ``[0, 1)``.
        """
        if not (0.0 <= beta < 1.0):
            raise ValueError("beta must be in range [0, 1)")

        super().__init__(params, lr)
        self.beta: float = beta
        self.v: list[NDArray[np.float64]] = []

        for param in self.params:
            v = np.zeros_like(param.data, dtype=np.float64)
            self.v.append(v)

    def step(self) -> None:
        """
        Делает один шаг с моментом.

        Для каждого параметра с непустым градиентом:

        - обновляет буфер in-place: ``v = beta * v + grad``;
        - обновляет параметр in-place: ``p.data -= lr * v``.

        Параметры с ``grad is None`` пропускаются — их буфер
        не обновляется, значение не меняется.
        """
        for i in range(len(self.params)):
            param = self.params[i]
            if param.grad is None:
                continue

            v = self.v[i]
            v *= self.beta
            v += param.grad
            
            param.data -= self.lr * v
