"""
Контейнер Sequential для композиции модулей.

Позволяет объединять слои, активации и другие модули в единую
последовательную модель. Прямой проход применяет модули в порядке
добавления, обратный — в обратном порядке.

Пример:

    >>> model = Sequential(
    ...     Linear(4, 8),
    ...     ReLU(),
    ...     Linear(8, 2),
    ... )
    >>> x = np.random.randn(5, 4)
    >>> y = model(x)
    >>> y.shape
    (5, 2)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .core import Module, Parameter


class Sequential(Module):
    """
    Последовательный контейнер модулей.

    Хранит упорядоченный список модулей и применяет их друг за другом.
    Параметров у самого контейнера нет — все параметры берутся из
    вложенных слоёв. Так же, как в PyTorch, ``Sequential`` не знает,
    что именно внутри: ``Linear``, ``ReLU`` или другой ``Sequential``.

    Позволяет обращаться к слоям по индексу (``model[0]``) и получать
    их число (``len(model)``), что удобно для отладки.

    Attributes:
        layers: Список модулей в порядке применения. Каждый элемент —
            наследник :class:`Module`.

    Example:
        >>> model = Sequential(Linear(4, 8), ReLU(), Linear(8, 2))
        >>> len(model)
        3
        >>> model[0]
        <numpy_nn.nn.layers.Linear object at 0x...>
        >>> x = np.random.randn(5, 4)
        >>> model(x).shape
        (5, 2)
    """

    def __init__(self, *layers: Module) -> None:
        """
        Создаёт контейнер из переданных модулей.

        Args:
            *layers: Модули в порядке применения. Может быть пусто —
                тогда ``Sequential`` ведёт себя как identity: ``forward``
                возвращает вход без изменений.
        """
        super().__init__()
        self.layers: list[Module] = list(layers)

    def forward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Применяет модули друг за другом в прямом порядке.

        Args:
            x: Входной массив. Форма должна соответствовать ожиданиям
                первого слоя.

        Returns:
            Выход последнего слоя.
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self, dout: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Прогоняет градиент через модули в обратном порядке.

        Каждый слой вызывает свой ``backward`` и возвращает градиент
        по входу, который становится ``dout`` для предыдущего слоя.

        Args:
            dout: Градиент функции потерь по выходу последнего слоя.

        Returns:
            Градиент функции потерь по входу первого слоя.
        """
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def parameters(self) -> list[Parameter]:
        """
        Собирает параметры всех вложенных модулей в плоский список.

        Порядок — сначала параметры первого слоя, потом второго и так
        далее. Модули без параметров (активации) не добавляют ничего.

        Returns:
            Плоский список :class:`Parameter` со всех слоёв.
        """
        p: list[Parameter] = []
        for layer in self.layers:
            p += layer.parameters()
        return p

    def __getitem__(self, index: int) -> Module:
        """
        Возвращает слой по индексу.

        Args:
            index: Позиция слоя (может быть отрицательной, как в списках).

        Returns:
            Модуль на данной позиции.
        """
        return self.layers[index]

    def __len__(self) -> int:
        """Возвращает число слоёв в контейнере."""
        return len(self.layers)
