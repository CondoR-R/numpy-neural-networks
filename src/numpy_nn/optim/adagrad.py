"""
Оптимизатор AdaGrad.

Идея: сделать learning rate адаптивным по каждому параметру. Для
каждого параметра накапливается сумма квадратов его градиентов,
и эффективный шаг делится на корень из этой суммы:

    G = G + grad**2
    p = p - lr * grad / (sqrt(G) + eps)

Параметры с большими градиентами получают маленький эффективный шаг,
с маленькими — большой. Это особенно полезно для разреженных признаков.

Главный недостаток: ``G`` монотонно растёт, поэтому эффективный
learning rate монотонно убывает и к концу обучения становится почти
нулевым. Именно из-за этого появились RMSProp (заменяет сумму на EMA)
и Adam (комбинирует EMA и momentum).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from numpy_nn.nn.core import Parameter

from .base import Optimizer


class AdaGrad(Optimizer):
    """
    Адаптивный градиентный спуск.

    Хранит для каждого параметра буфер накопленных квадратов градиентов
    той же формы. На каждом шаге буфер увеличивается на ``grad**2``,
    а параметр обновляется с эффективным learning rate, обратно
    пропорциональным корню из буфера.

    Параметры с ``grad is None`` пропускаются: их буфер не обновляется,
    значение не меняется.

    Attributes:
        params: Список :class:`Parameter` из базового класса.
        lr: Начальный learning rate. По умолчанию ``0.01`` — AdaGrad
            обычно требует большего ``lr``, чем SGD, потому что
            эффективный шаг уменьшается делением на ``sqrt(G)``.
        eps: Малая константа для численной устойчивости, добавляется
            к ``sqrt(G)`` перед делением. По умолчанию ``1e-8``.
        g: Список буферов накопленных квадратов градиентов, по одному
            на каждый параметр. Форма каждого совпадает с формой
            соответствующего ``param.data``.

    Example:
        >>> from numpy_nn.nn import Linear
        >>> layer = Linear(4, 3)
        >>> optimizer = AdaGrad(layer.parameters(), lr=0.01)
        >>> optimizer.zero_grad()
        >>> # ... forward, backward ...
        >>> optimizer.step()
    """

    def __init__(
        self,
        params: list[Parameter],
        lr: float = 0.01,
        eps: float = 1e-8,
    ) -> None:
        """
        Инициализирует оптимизатор AdaGrad.

        Args:
            params: Параметры для обновления.
            lr: Скорость обучения, должна быть больше 0.
            eps: Константа для численной устойчивости, должна быть
                больше 0.

        Raises:
            ValueError: Если ``lr`` или ``eps`` не положительны.
        """
        if not (eps > 0):
            raise ValueError("eps must be more than 0")

        super().__init__(params, lr)
        self.eps: float = eps
        self.g: list[NDArray[np.float64]] = []

        for param in self.params:
            g = np.zeros_like(param.data, dtype=np.float64)
            self.g.append(g)

    def step(self) -> None:
        """
        Делает один шаг AdaGrad.

        Для каждого параметра с непустым градиентом:

        - обновляет буфер in-place: ``g = g + grad**2``;
        - обновляет параметр: ``p.data -= lr * grad / (sqrt(g) + eps)``.

        Параметры с ``grad is None`` пропускаются.
        """
        for i in range(len(self.params)):
            param = self.params[i]
            if param.grad is None:
                continue

            g = self.g[i]
            g += param.grad ** 2
            param.data -= self.lr * param.grad / (np.sqrt(g) + self.eps)
