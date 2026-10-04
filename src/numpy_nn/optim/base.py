"""
Общий интерфейс оптимизаторов.

Модуль задаёт базовый класс :class:`Optimizer`, от которого наследуются
все конкретные оптимизаторы библиотеки: SGD, Momentum, Adam и другие.

Единый интерфейс нужен, чтобы работа с любым оптимизатором выглядела
одинаково:

- создать с параметрами и learning rate;
- вызвать :meth:`Optimizer.step` для обновления параметров;
- вызвать :meth:`Optimizer.zero_grad` для обнуления градиентов.

Базовый класс не знает ни одной конкретной формулы обновления — только
общую структуру. Формулы реализуются в наследниках.
"""

from __future__ import annotations

from numpy_nn.nn.core import Parameter


class Optimizer:
    """
    Базовый класс оптимизаторов.

    Задаёт единый интерфейс: :meth:`step` и :meth:`zero_grad`. Наследники
    реализуют :meth:`step` под конкретный алгоритм обновления.

    Оптимизатор не знает про модели и слои — он работает только со
    списком :class:`Parameter`. Это позволяет обучать как всю модель
    целиком, так и её часть, передавая в оптимизатор разные наборы
    параметров.

    Attributes:
        params: Список :class:`Parameter`, которые оптимизатор обновляет.
            Хранится копия переданного списка — внешние изменения
            не влияют на оптимизатор.
        lr: Learning rate (скорость обучения).

    Example:
        >>> from numpy_nn.nn import Linear
        >>> from numpy_nn.optim import SGD
        >>> layer = Linear(4, 3)
        >>> optimizer = SGD(layer.parameters(), lr=0.01)
        >>> optimizer.zero_grad()
        >>> # ... forward, backward ...
        >>> optimizer.step()
    """

    def __init__(self, params: list[Parameter], lr: float) -> None:
        """
        Инициализирует оптимизатор.

        Args:
            params: Параметры, которые оптимизатор будет обновлять.
                Сохраняется копия списка.
            lr: Скорость обучения.
        """
        if not (lr > 0):
            raise ValueError("learning rate must be more than 0")
        self.params: list[Parameter] = list(params)
        self.lr: float = lr

    def step(self) -> None:
        """
        Один шаг оптимизации.

        Абстрактный метод. Должен быть реализован в наследнике под
        конкретный алгоритм (SGD, Momentum, Adam и т.д.). Ничего
        не возвращает — обновляет параметры in-place.

        Raises:
            NotImplementedError: Если наследник не реализовал метод.
        """
        raise NotImplementedError(
            "step должен быть реализован в наследнике, иначе ничего не работает"
        )

    def zero_grad(self) -> None:
        """
        Обнуляет градиенты параметров, выставляя ``grad = None``.

        Обходит :attr:`params` и вызывает :meth:`Parameter.zero_grad`
        у каждого. Обнуляются градиенты только тех параметров, которые
        обучает данный оптимизатор — что важно, когда часть модели
        заморожена и не передана в оптимизатор.
        """
        for p in self.params:
            p.zero_grad()
