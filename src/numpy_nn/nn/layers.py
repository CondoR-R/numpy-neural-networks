"""
Полносвязные слои.

Модуль содержит класс :class:`Linear` — базовый строительный блок
многослойного перцептрона.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .core import Module, Parameter


class Linear(Module):
    """
    Полносвязный слой.

    Вычисляет ``Y = X @ W.T + b``, где ``W`` имеет форму
    ``(out_features, in_features)`` — та же конвенция, что в PyTorch.

    Веса инициализируются по Xavier/Glorot (равномерное распределение
    на ``[-limit, limit]``, ``limit = sqrt(6 / (in_features + out_features))``),
    смещения — нулями.

    Attributes:
        in_features: Число входных признаков.
        out_features: Число выходных признаков.
        weight: Параметр формы ``(out_features, in_features)``.
        bias: Параметр формы ``(out_features,)``.
        x: Сохранённый вход последнего ``forward`` — нужен для
            ``backward``. ``None`` до первого вызова.

    Example:
        >>> layer = Linear(4, 3)
        >>> x = np.random.randn(5, 4)
        >>> y = layer(x)
        >>> y.shape
        (5, 3)
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        """
        Создаёт слой и инициализирует веса.

        Args:
            in_features: Размер входного признакового пространства.
            out_features: Число нейронов в слое.
        """
        super().__init__()
        self.in_features: int = in_features
        self.out_features: int = out_features
        limit = np.sqrt(6 / (out_features + in_features))
        self.weight: Parameter = Parameter(
            np.random.uniform(
                -limit,
                limit,
                size=(out_features, in_features),
                
            ).astype(np.float64)
        )
        self.bias: Parameter = Parameter(np.zeros(out_features, dtype=np.float64))
        self.x: NDArray[np.float64] | None = None

    def forward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Считает линейное преобразование ``x @ W.T + b``.

        Сохраняет вход в ``self.x`` для использования в :meth:`backward`.
        Копия не делается: вход не мутируется, а хранится только ссылка.

        Args:
            x: Вход формы ``(N, in_features)`` или ``(in_features,)``
                для одиночного примера.

        Returns:
            Выход формы ``(N, out_features)`` или ``(out_features,)``.
        """
        self.x = x
        return x @ self.weight.data.T + self.bias.data

    def backward(self, dout: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Считает градиенты по параметрам и по входу.

        Формулы (для ``Y = X @ W.T + b``):

        - ``dW = dout.T @ X``  — форма ``(out, in)``, как ``W``
        - ``db = dout.sum(axis=0)``  — форма ``(out,)``, как ``b``
        - ``dX = dout @ W``  — форма ``(N, in)``, для предыдущего слоя

        Градиенты записываются в ``weight.grad`` и ``bias.grad``.
        Градиент по входу возвращается наружу.

        Args:
            dout: Градиент функции потерь по выходу слоя,
                формы ``(N, out_features)``.

        Returns:
            Градиент по входу формы ``(N, in_features)``.

        Raises:
            AssertionError: Если :meth:`forward` не был вызван до backward —
                тогда ``self.x`` не сохранён.
        """
        assert self.x is not None, "forward must be called before backward"
        self.weight.grad = dout.T @ self.x
        self.bias.grad = dout.sum(axis=0)
        dx = dout @ self.weight.data
        return dx

    def parameters(self) -> list[Parameter]:
        """
        Возвращает ``[weight, bias]`` в этом порядке.

        Returns:
            Список из двух :class:`Parameter`.
        """
        return [self.weight, self.bias]
