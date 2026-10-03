"""
Экспоненциальное скользящее среднее.

Утилита для сглаживания последовательностей чисел или массивов
с фиксированной формой. Каждое новое значение смешивается с текущим
сглаженным состоянием с коэффициентом ``beta``:

    s_t = beta * s_{t-1} + (1 - beta) * x_t

Применяется с bias correction, чтобы первые значения не были смещены
к нулю из-за инициализации ``s_0 = 0``:

    s_hat_t = s_t / (1 - beta^t)

Пример использования для сглаживания метрики:

    >>> ema = EMA(beta=0.9)
    >>> for loss in [2.0, 1.8, 1.5, 1.3]:
    ...     ema.update(loss)
    >>> ema.value
    array(...)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class EMA:
    """
    Экспоненциальное скользящее среднее.

    Хранит сглаженное состояние и обновляет его по формуле
    ``s = beta * s + (1 - beta) * x``. Bias correction применяется
    автоматически при чтении :attr:`value`.

    Состояние инициализируется нулями в первом вызове :meth:`update`,
    чтобы bias correction была корректной. Это стандартный подход
    (так же устроены моменты в Adam).

    Работает и со скалярами, и с массивами любой формы. Форма
    фиксируется при первом обновлении и должна сохраняться — иначе
    бросается :class:`ValueError`.

    Attributes:
        beta: Коэффициент сглаживания в диапазоне ``[0, 1)``. Чем
            ближе к 1, тем сильнее сглаживание и больше эффективное
            окно усреднения.
        s: Текущее сглаженное состояние без bias correction. ``None``
            до первого вызова :meth:`update`.
        t: Число выполненных обновлений. Используется для bias
            correction в :attr:`value`.

    Example:
        >>> ema = EMA(beta=0.9)
        >>> ema.update(np.array([1.0, 2.0, 3.0]))
        >>> ema.update(np.array([2.0, 2.0, 3.0]))
        >>> ema.value.shape
        (3,)
    """

    def __init__(self, beta: float = 0.9) -> None:
        """
        Создаёт объект EMA с заданным коэффициентом.

        Args:
            beta: Коэффициент сглаживания. Должен лежать в диапазоне
                ``[0, 1)``.

        Raises:
            ValueError: Если ``beta`` вне диапазона ``[0, 1)``.
        """
        if not (0.0 <= beta < 1.0):
            raise ValueError("beta must be in range [0, 1)")

        self.beta: float = beta
        self.s: NDArray[np.float64] | None = None
        self.t: int = 0

    def update(self, x: NDArray[np.float64] | float) -> None:
        """
        Обновляет сглаженное состояние новым значением.

        При первом вызове состояние инициализируется нулями той же
        формы, что ``x``. Далее применяется формула EMA in-place.

        Args:
            x: Новое значение — скаляр или массив формы, совпадающей
                с формой предыдущих значений.

        Raises:
            ValueError: Если форма ``x`` не совпадает с формой
                предыдущих обновлений.
        """
        x = np.asarray(x, dtype=np.float64)

        if self.s is None:
            self.s = np.zeros_like(x, dtype=np.float64)

        if self.s.shape != x.shape:
            raise ValueError(
                f"shape changed from {self.s.shape} to {x.shape}"
            )

        self.s *= self.beta
        self.s += (1.0 - self.beta) * x
        self.t += 1

    @property
    def value(self) -> NDArray[np.float64]:
        """
        Текущее значение EMA с bias correction.

        Возвращает ``s / (1 - beta^t)``. Bias correction компенсирует
        смещение первых значений, вызванное инициализацией нулями.

        Returns:
            Сглаженное значение той же формы, что и входные данные.

        Raises:
            RuntimeError: Если :meth:`update` ни разу не вызывался.
        """
        if self.s is None:
            raise RuntimeError("EMA value requested before any update")
        return self.s / (1.0 - self.beta ** self.t)
