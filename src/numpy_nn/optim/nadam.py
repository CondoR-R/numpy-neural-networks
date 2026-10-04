"""
Оптимизатор Nadam.

Комбинация Adam и Nesterov momentum. От Adam отличается только
числителем обновления: вместо чистого ``m_hat`` используется
Nesterov-слагаемое, которое добавляет к сглаженному моменту
«свежую» часть градиента с bias correction по следующему шагу.

Формула (классическая форма Dozat, 2016):

    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad**2
    m_hat = m / (1 - beta1**t)
    m_nesterov = beta1 * m_hat + (1 - beta1) * grad / (1 - beta1**(t+1))
    v_hat = v / (1 - beta2**t)
    p = p - lr * m_nesterov / (sqrt(v_hat) + eps)

Смысл Nesterov-слагаемого — тот же, что в NAG по сравнению с Momentum:
оптимизатор быстрее реагирует на смену направления градиента.
"""

from __future__ import annotations

import numpy as np

from numpy_nn.nn.core import Parameter

from .adam import Adam


class Nadam(Adam):
    """
    Nesterov-accelerated Adam.

    Наследует от :class:`Adam` конструктор, валидацию гиперпараметров
    и поля состояния (``m``, ``v``, ``t``). Переопределяет только
    :meth:`step`: обновление буферов и bias correction второго момента
    остаются как в Adam, а вместо ``m_hat`` используется
    Nesterov-комбинация ``beta1 * m_hat + (1 - beta1) * grad / bc1_next``.

    Параметры с ``grad is None`` пропускаются, ``t`` растёт независимо
    от наличия градиентов — как в Adam.

    Attributes:
        params: Список :class:`Parameter` (унаследовано).
        lr: Learning rate (унаследовано).
        beta1: Сглаживание первого момента (унаследовано).
        beta2: Сглаживание второго момента (унаследовано).
        eps: Константа для численной устойчивости (унаследовано).
        t: Счётчик шагов (унаследовано).
        m: Буферы первого момента (унаследовано).
        v: Буферы второго момента (унаследовано).

    Example:
        >>> from numpy_nn.nn import Linear
        >>> layer = Linear(4, 3)
        >>> optimizer = Nadam(layer.parameters(), lr=0.001)
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
        Инициализирует оптимизатор Nadam.

        Args:
            params: Параметры для обновления.
            lr: Скорость обучения, неотрицательна.
            beta1: Коэффициент сглаживания первого момента, ``[0, 1)``.
            beta2: Коэффициент сглаживания второго момента, ``[0, 1)``.
            eps: Константа для численной устойчивости, больше 0.

        Raises:
            ValueError: Если гиперпараметры вне допустимых диапазонов
                (проверяется в :class:`Adam`).
        """
        super().__init__(params, lr, beta1, beta2, eps)

    def step(self) -> None:
        """
        Делает один шаг Nadam.

        Увеличивает ``t``, считает три bias-correction коэффициента
        один раз на весь шаг, затем для каждого параметра с непустым
        градиентом:

        - обновляет первый момент: ``m = beta1*m + (1-beta1)*grad``;
        - обновляет второй момент: ``v = beta2*v + (1-beta2)*grad**2``;
        - считает ``m_hat = m / (1 - beta1**t)``;
        - считает Nesterov-слагаемое:
          ``beta1 * m_hat + (1 - beta1) * grad / (1 - beta1**(t+1))``;
        - обновляет параметр:
          ``p -= lr * m_nesterov / (sqrt(v / (1 - beta2**t)) + eps)``.

        Параметры с ``grad is None`` пропускаются.
        """
        self.t += 1
        bc1 = 1 - self.beta1**self.t
        bc2 = 1 - self.beta2**self.t
        bc1_next = 1 - self.beta1 ** (self.t + 1)

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
            m_nesterov = (
                self.beta1 * m_hat
                + (1 - self.beta1) * param.grad / bc1_next
            )

            param.data -= self.lr * m_nesterov / (np.sqrt(v / bc2) + self.eps)
