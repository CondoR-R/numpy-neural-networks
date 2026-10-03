"""
Функции потерь.

Содержит две стандартные функции потерь:

- :class:`CrossEntropyLoss` — для многоклассовой классификации.
  Принимает логиты и целочисленные метки, делает softmax внутри.
- :class:`MSELoss` — для регрессии и любых задач с непрерывным выходом.

Оба класса **не наследуются** от :class:`Module`: их сигнатуры
несовместимы с базовым интерфейсом (``forward`` принимает два аргумента,
``backward`` не принимает ничего). Loss — не слой, а терминальная
функция в вычислительной цепочке.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class CrossEntropyLoss:
    """
    Кросс-энтропия для многоклассовой классификации.

    Принимает **логиты** (сырые выходы модели до softmax) и **метки
    классов** целыми числами. Внутри делает численно устойчивый
    softmax, затем считает средний отрицательный логарифм вероятности
    правильного класса по батчу.

    Поле ``probs`` сохраняется для использования в :meth:`backward`,
    где градиент по логитам вычисляется как ``(probs - one_hot) / N``.

    Attributes:
        probs: Вероятности после softmax формы ``(N, C)``. ``None``
            до первого вызова :meth:`forward`.
        target: Сохранённые метки формы ``(N,)`` и dtype ``int64``.
            ``None`` до первого вызова.
        n: Число примеров в батче. ``None`` до первого вызова.
            Кэшируется, чтобы не считать ``shape[0]`` повторно
            в :meth:`backward`.

    Example:
        >>> criterion = CrossEntropyLoss()
        >>> logits = np.array([[1.0, 2.0, 3.0], [1.0, 0.0, -1.0]])
        >>> target = np.array([2, 0])
        >>> loss = criterion(logits, target)
        >>> dlogits = criterion.backward()
        >>> dlogits.shape
        (2, 3)
    """

    def __init__(self) -> None:
        """Создаёт объект функции потерь без параметров."""
        self.probs: NDArray[np.float64] | None = None
        self.target: NDArray[np.int64] | None = None
        self.n: int | None = None

    def __call__(
        self,
        logits: NDArray[np.float64],
        target: NDArray[np.int64],
    ) -> float:
        """
        Делегирует в :meth:`forward`.

        Позволяет писать ``criterion(logits, target)`` как в PyTorch.

        Args:
            logits: Логиты формы ``(N, C)``.
            target: Метки формы ``(N,)``.

        Returns:
            Скалярный loss.
        """
        return self.forward(logits, target)

    def forward(
        self,
        logits: NDArray[np.float64],
        target: NDArray[np.int64],
    ) -> float:
        """
        Считает среднюю кросс-энтропию по батчу.

        Численно устойчивый softmax: из логитов вычитается максимум
        по оси классов. Это не меняет математический результат
        (softmax инвариантен к сдвигу по классам), но убирает
        переполнение ``exp`` при больших логитах.

        Loss усредняется по батчу: ``-mean(log p[y])``.

        Args:
            logits: Логиты формы ``(N, C)``.
            target: Целочисленные метки формы ``(N,)`` со значениями
                в диапазоне ``[0, C)``.

        Returns:
            Скалярный loss. Для идеальных предсказаний стремится к 0;
            для случайного угадывания близко к ``log(C)``.
        """
        m = logits.max(axis=1, keepdims=True)
        shifted = logits - m
        exp = np.exp(shifted)
        probs = exp / exp.sum(axis=1, keepdims=True)
        n = target.shape[0]

        self.probs = probs
        self.target = target
        self.n = n

        loss = -np.log(probs[np.arange(n), target]).mean()
        return float(loss)

    def backward(self) -> NDArray[np.float64]:
        """
        Градиент loss по логитам.

        Формула: ``(probs - one_hot(target)) / N``. Единица вычитается
        в позиции правильного класса для каждого примера, затем
        результат делится на размер батча — потому что loss усреднён.

        Returns:
            Массив формы ``(N, C)`` — градиент по логитам, поданным
            в последний :meth:`forward`.

        Raises:
            AssertionError: Если :meth:`forward` не был вызван раньше.
        """
        assert self.probs is not None, "forward must be called before backward"
        assert self.target is not None, "forward must be called before backward"
        assert self.n is not None, "forward must be called before backward"

        dlogits = self.probs.copy()
        dlogits[np.arange(self.n), self.target] -= 1.0
        dlogits /= self.n
        return dlogits


class MSELoss:
    """
    Среднеквадратичная ошибка (Mean Squared Error).

    Считает среднее квадратов разностей между предсказанием и истиной
    по всем элементам массивов. Форма входа произвольная, важно лишь,
    чтобы ``pred`` и ``target`` совпадали по форме.

    Attributes:
        diff: Сохранённая разность ``pred - target``. ``None`` до
            первого вызова :meth:`forward`.
        n: Общее число элементов в ``diff``. ``None`` до первого
            вызова. Кэшируется для :meth:`backward`.

    Example:
        >>> criterion = MSELoss()
        >>> pred = np.array([1.0, 2.0, 3.0])
        >>> target = np.array([1.5, 2.0, 2.5])
        >>> loss = criterion(pred, target)
        >>> dpred = criterion.backward()
        >>> dpred.shape
        (3,)
    """

    def __init__(self) -> None:
        """Создаёт объект функции потерь без параметров."""
        self.diff: NDArray[np.float64] | None = None
        self.n: int | None = None

    def __call__(
        self,
        pred: NDArray[np.float64],
        target: NDArray[np.float64],
    ) -> float:
        """
        Делегирует в :meth:`forward`.

        Позволяет писать ``criterion(pred, target)``.

        Args:
            pred: Предсказание произвольной формы.
            target: Истинное значение той же формы.

        Returns:
            Скалярный loss.
        """
        return self.forward(pred, target)

    def forward(
        self,
        pred: NDArray[np.float64],
        target: NDArray[np.float64],
    ) -> float:
        """
        Считает MSE: среднее ``(pred - target)**2`` по всем элементам.

        Args:
            pred: Предсказание произвольной формы.
            target: Истинное значение той же формы.

        Returns:
            Скалярный loss — среднее квадратов разностей. Ноль при
            идеальном совпадении, всегда неотрицателен.
        """
        self.diff = pred - target
        self.n = self.diff.size

        return float((self.diff ** 2).mean())

    def backward(self) -> NDArray[np.float64]:
        """
        Градиент MSE по ``pred``.

        Формула: ``2 * (pred - target) / n``, где ``n`` — общее число
        элементов в массиве (не размер батча, а именно ``diff.size``).

        Returns:
            Массив той же формы, что ``pred``.

        Raises:
            AssertionError: Если :meth:`forward` не был вызван раньше.
        """
        assert self.diff is not None, "forward must be called before backward"
        assert self.n is not None, "forward must be called before backward"

        return (2.0 / self.n) * self.diff
