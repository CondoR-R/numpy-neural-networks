"""
Оптимизатор RMSProp.

Идея: сделать learning rate адаптивным по каждому параметру, но без
монотонного затухания, как в AdaGrad. Вместо суммы квадратов
градиентов используется их экспоненциальное скользящее среднее:

    v = beta * v + grad**2
    p = p - lr * grad / (sqrt(v) + eps)

Параметры с большими недавними градиентами получают маленький
эффективный шаг, с маленькими — большой. Старые градиенты забываются
экспоненциально, поэтому эффективный learning rate не стремится к нулю,
а стабилизируется около разумного значения.

Отличие от AdaGrad — только в формуле обновления буфера: ``v`` вместо
``G``, EMA вместо суммы. Отсюда и главное преимущество: RMSProp хорошо
работает на нестационарных задачах (RNN, меняющиеся распределения),
где AdaGrad быстро замедляется.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from numpy_nn.nn.core import Parameter

from .base import Optimizer


class RMSProp(Optimizer):
    """
    Root Mean Square Propagation.

    Хранит для каждого параметра буфер экспоненциально сглаженных
    квадратов градиентов той же формы. На каждом шаге буфер
    обновляется, а параметр делится на корень из буфера — так
    эффективный learning rate адаптируется к каждому параметру.

    Параметры с ``grad is None`` пропускаются: их буфер не обновляется,
    значение не меняется.

    Attributes:
        params: Список :class:`Parameter` из базового класса.
        lr: Начальный learning rate. По умолчанию ``0.01``.
        beta: Коэффициент сглаживания буфера. Обычно ``0.9``.
            Значение должно быть в диапазоне ``[0, 1)``.
        eps: Малая константа для численной устойчивости, добавляется
            к ``sqrt(v)`` перед делением. По умолчанию ``1e-8``.
        v: Список буферов сглаженных квадратов градиентов, по одному
            на каждый параметр. Форма каждого совпадает с формой
            соответствующего ``param.data``.

    Example:
        >>> from numpy_nn.nn import Linear
        >>> layer = Linear(4, 3)
        >>> optimizer = RMSProp(layer.parameters(), lr=0.01, beta=0.9)
        >>> optimizer.zero_grad()
        >>> # ... forward, backward ...
        >>> optimizer.step()
    """

    def __init__(
        self,
        params: list[Parameter],
        lr: float = 0.01,
        beta: float = 0.9,
        eps: float = 1e-8,
    ) -> None:
        """
        Инициализирует оптимизатор RMSProp.

        Args:
            params: Параметры для обновления.
            lr: Скорость обучения, должна быть неотрицательной
                (проверяется в базовом классе).
            beta: Коэффициент сглаживания EMA, в диапазоне ``[0, 1)``.
            eps: Константа для численной устойчивости, больше 0.

        Raises:
            ValueError: Если ``beta`` вне диапазона ``[0, 1)`` или
                ``eps <= 0``.
        """
        if not (0.0 <= beta < 1.0):
            raise ValueError("beta must be in range [0, 1)")
        if eps <= 0:
            raise ValueError("eps must be more than 0")

        super().__init__(params, lr)
        self.beta: float = beta
        self.eps: float = eps

        self.v: list[NDArray[np.float64]] = []
        for param in self.params:
            v = np.zeros_like(param.data, dtype=np.float64)
            self.v.append(v)

    def step(self) -> None:
        """
        Делает один шаг RMSProp.

        Для каждого параметра с непустым градиентом:

        - обновляет буфер in-place:
          ``v = beta * v + grad**2``;
        - обновляет параметр:
          ``p.data -= lr * grad / (sqrt(v) + eps)``.

        Параметры с ``grad is None`` пропускаются.
        """
        for i in range(len(self.params)):
            param = self.params[i]
            if param.grad is None:
                continue

            v = self.v[i]
            v *= self.beta
            v += param.grad**2

            param.data -= self.lr * param.grad / (np.sqrt(v) + self.eps)
