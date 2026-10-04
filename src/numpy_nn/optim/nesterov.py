"""
Оптимизатор Nesterov Accelerated Gradient (NAG).

Улучшение Momentum: вместо шага по накопленной скорости NAG двигается
по «эффективному градиенту», который учитывает и текущий градиент,
и momentum. Это даёт более быструю реакцию на изменение направления
градиента.

Формула (в стиле PyTorch):

    v = beta * v + grad
    d = grad + beta * v
    p = p - lr * d

Отличие от Momentum — только в последнем шаге: шаг делается по ``d``,
а не по ``v``. Обрати внимание, что ``d`` использует **уже обновлённый**
``v`` — так PyTorch реализует эквивалентную формулу Nesterov без второго
forward pass.
"""

from __future__ import annotations

from numpy_nn.nn.core import Parameter

from .momentum import Momentum


class Nesterov(Momentum):
    """
    Стохастический градиентный спуск с моментом Нестерова.

    Наследует от :class:`Momentum` валидацию ``beta``, инициализацию
    буфера скорости и поля ``params``, ``lr``, ``beta``, ``v``.
    Переопределяет только :meth:`step` — порядок вычислений отличается
    на одну строку.

    На первом шаге, когда ``v = 0``, эффективный градиент
    ``d = grad + beta * grad = (1 + beta) * grad``, то есть шаг
    получается больше, чем у Momentum. Это не ошибка — это способ
    NAG «двигаться увереннее» в выбранном направлении.

    Attributes:
        params: Список :class:`Parameter` (унаследовано).
        lr: Learning rate (унаследовано).
        beta: Коэффициент сглаживания буфера (унаследовано).
        v: Список буферов скорости (унаследовано).

    Example:
        >>> from numpy_nn.nn import Linear
        >>> layer = Linear(4, 3)
        >>> optimizer = Nesterov(layer.parameters(), lr=0.01, beta=0.9)
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
        Инициализирует оптимизатор Nesterov.

        Args:
            params: Параметры для обновления.
            lr: Скорость обучения.
            beta: Коэффициент сглаживания буфера скорости.

        Raises:
            ValueError: Если ``beta`` вне диапазона ``[0, 1)``.
        """
        super().__init__(params, lr, beta)

    def step(self) -> None:
        """
        Делает один шаг NAG.

        Для каждого параметра с непустым градиентом:

        - обновляет буфер: ``v = beta * v + grad``;
        - считает эффективный градиент: ``d = grad + beta * v``;
        - обновляет параметр: ``p.data -= lr * d``.

        Параметры с ``grad is None`` пропускаются.
        """
        for i in range(len(self.params)):
            param = self.params[i]
            if param.grad is None:
                continue

            v = self.v[i]

            v *= self.beta
            v += param.grad

            d = param.grad + self.beta * v
            param.data -= self.lr * d
