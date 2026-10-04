"""
Оптимизатор Adam (Adaptive Moment Estimation).

Комбинирует две идеи:

- Momentum — сглаживание самих градиентов (первый момент ``m``);
- RMSProp — сглаживание квадратов градиентов (второй момент ``v``).

Дополнительно применяется **bias correction** — поправка на смещение
EMA в начале обучения, потому что оба буфера инициализируются нулями
и без неё первые шаги сильно занижены.

Формула (вариант из статьи Kingma & Ba, 2014):

    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad**2
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    p = p - lr * m_hat / (sqrt(v_hat) + eps)

где ``t`` — номер шага, начиная с 1.

Обрати внимание: Adam использует **классическую формулу EMA**
(с множителем ``(1 - beta)`` перед новым значением), в отличие от
RMSProp, где применяется PyTorch-стиль ``v = beta*v + grad**2``.
Без этой формы bias correction не имеет математического смысла.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from numpy_nn.nn.core import Parameter

from .base import Optimizer


class Adam(Optimizer):
    """
    Adam — адаптивная оценка моментов.

    Хранит для каждого параметра два буфера той же формы: первый
    момент ``m`` (сглаженные градиенты) и второй момент ``v``
    (сглаженные квадраты градиентов). На каждом шаге оба буфера
    обновляются, применяется bias correction, и параметр двигается
    по нормализованному первому моменту.

    Параметры с ``grad is None`` пропускаются: их буферы не обновляются,
    значения не меняются. Счётчик шагов ``t`` при этом всё равно растёт,
    потому что он относится к вызову :meth:`step`, а не к каждому
    отдельному параметру.

    Attributes:
        params: Список :class:`Parameter` из базового класса.
        lr: Начальный learning rate. По умолчанию ``0.001``.
        beta1: Коэффициент сглаживания первого момента. Обычно ``0.9``.
        beta2: Коэффициент сглаживания второго момента. Обычно ``0.999``.
        eps: Малая константа для численной устойчивости.
            По умолчанию ``1e-8``.
        t: Число выполненных шагов. Используется для bias correction.
        m: Список буферов первого момента, по одному на параметр.
        v: Список буферов второго момента, по одному на параметр.

    Example:
        >>> from numpy_nn.nn import Linear
        >>> layer = Linear(4, 3)
        >>> optimizer = Adam(layer.parameters(), lr=0.001)
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
    ) -> None:
        """
        Инициализирует оптимизатор Adam.

        Args:
            params: Параметры для обновления.
            lr: Скорость обучения, должна быть неотрицательной
                (проверяется в базовом классе).
            beta1: Коэффициент сглаживания первого момента, в ``[0, 1)``.
            beta2: Коэффициент сглаживания второго момента, в ``[0, 1)``.
            eps: Константа для численной устойчивости, больше 0.

        Raises:
            ValueError: Если ``beta1`` или ``beta2`` вне ``[0, 1)``,
                либо ``eps <= 0``.
        """
        if not (0.0 <= beta1 < 1.0):
            raise ValueError("beta1 must be in range [0, 1)")
        if not (0.0 <= beta2 < 1.0):
            raise ValueError("beta2 must be in range [0, 1)")
        if not (eps > 0):
            raise ValueError("eps must be more than 0")

        super().__init__(params, lr)
        self.beta1: float = beta1
        self.beta2: float = beta2
        self.eps: float = eps

        self.t: int = 0
        self.m: list[NDArray[np.float64]] = []
        self.v: list[NDArray[np.float64]] = []
        for param in self.params:
            self.m.append(np.zeros_like(param.data, dtype=np.float64))
            self.v.append(np.zeros_like(param.data, dtype=np.float64))

    def step(self) -> None:
        """
        Делает один шаг Adam.

        Увеличивает счётчик ``t``, считает поправки bias correction
        один раз на весь шаг, затем для каждого параметра с непустым
        градиентом:

        - обновляет первый момент: ``m = beta1*m + (1-beta1)*grad``;
        - обновляет второй момент: ``v = beta2*v + (1-beta2)*grad**2``;
        - считает ``m_hat = m / (1 - beta1**t)`` и
          ``v_hat = v / (1 - beta2**t)``;
        - обновляет параметр:
          ``p.data -= lr * m_hat / (sqrt(v_hat) + eps)``.

        Параметры с ``grad is None`` пропускаются.
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

            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
