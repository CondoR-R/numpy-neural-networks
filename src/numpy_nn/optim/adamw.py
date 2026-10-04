"""
Оптимизатор AdamW.

Модификация Adam с **decoupled weight decay** (Loshchilov & Hutter, 2017).
Отличие от Adam — в том, что штраф на веса применяется отдельно от
градиента, а не через добавление L2-слагаемого в loss.

Зачем это нужно. Если добавить L2-штраф ``lambda * p`` прямо в градиент,
то в адаптивных оптимизаторах он тоже поделится на ``sqrt(v_hat)``,
и для параметров с большими градиентами штраф ослабнет, а с маленькими
усилится. Это не то, чего мы хотим от регуляризации. AdamW выносит
weight decay **до** Adam-шага, и штраф действует одинаково на все
параметры.

Формула:

    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad**2
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    p = p * (1 - lr * weight_decay)      # decay первым
    p = p - lr * m_hat / (sqrt(v_hat) + eps)

Порядок операций совпадает с PyTorch: decay применяется до Adam-шага.
"""

from __future__ import annotations

import numpy as np

from numpy_nn.nn.core import Parameter

from .adam import Adam


class AdamW(Adam):
    """
    Adam с decoupled weight decay.

    Наследует от :class:`Adam` конструктор, валидацию гиперпараметров
    и поля состояния (``m``, ``v``, ``t``). Переопределяет только
    :meth:`step`, добавляя перед Adam-шагом умножение параметра
    на ``(1 - lr * weight_decay)``.

    Параметры с ``grad is None`` пропускаются целиком — их weight decay
    **не применяется**. Это важно: если параметр не участвовал
    в обучении, регуляризация его не должна трогать.

    Attributes:
        params: Список :class:`Parameter` (унаследовано от Adam).
        lr: Learning rate (унаследовано).
        beta1: Сглаживание первого момента (унаследовано).
        beta2: Сглаживание второго момента (унаследовано).
        eps: Константа для численной устойчивости (унаследовано).
        t: Счётчик шагов (унаследовано).
        m: Буферы первого момента (унаследовано).
        v: Буферы второго момента (унаследовано).
        weight_decay: Коэффициент регуляризации. Должен быть ``>= 0``.
            По умолчанию ``0.01``. Значение ``0`` отключает
            регуляризацию, и AdamW совпадает с Adam.

    Example:
        >>> from numpy_nn.nn import Linear
        >>> layer = Linear(4, 3)
        >>> optimizer = AdamW(layer.parameters(), lr=0.001, weight_decay=0.01)
        >>> optimizer.zero_grad()
        >>> # ... forward, backward ...
        >>> optimizer.step()
    """

    def __init__(
        self,
        params: list[Parameter],
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        """
        Инициализирует оптимизатор AdamW.

        Args:
            params: Параметры для обновления.
            lr: Скорость обучения, неотрицательна.
            beta1: Коэффициент сглаживания первого момента, ``[0, 1)``.
            beta2: Коэффициент сглаживания второго момента, ``[0, 1)``.
            eps: Константа для численной устойчивости, больше 0.
            weight_decay: Коэффициент регуляризации, ``>= 0``.

        Raises:
            ValueError: Если ``weight_decay < 0`` или параметры Adam
                вне допустимых диапазонов.
        """
        if not (weight_decay >= 0):
            raise ValueError("weight_decay must be more or equal than 0")

        super().__init__(params, lr, beta1, beta2, eps)
        self.weight_decay: float = weight_decay

    def step(self) -> None:
        """
        Делает один шаг AdamW.

        Увеличивает счётчик ``t``, считает поправки bias correction
        один раз на весь шаг, затем для каждого параметра с непустым
        градиентом:

        - обновляет первый момент: ``m = beta1*m + (1-beta1)*grad``;
        - обновляет второй момент: ``v = beta2*v + (1-beta2)*grad**2``;
        - считает ``m_hat`` и ``v_hat``;
        - применяет weight decay: ``p *= 1 - lr * weight_decay``;
        - обновляет параметр:
          ``p -= lr * m_hat / (sqrt(v_hat) + eps)``.

        Параметры с ``grad is None`` пропускаются: weight decay к ним
        не применяется.
        """
        self.t += 1
        bc1 = 1 - self.beta1**self.t
        bc2 = 1 - self.beta2**self.t

        for i in range(len(self.params)):
            param = self.params[i]
            if param.grad is None:
                continue

            m = self.m[i]
            v = self.v[i]

            m *= self.beta1
            m += (1 - self.beta1) * param.grad

            v *= self.beta2
            v += (1 - self.beta2) * param.grad**2

            m_hat = m / bc1
            v_hat = v / bc2

            param.data *= (1 - self.lr * self.weight_decay)
            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
