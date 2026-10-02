"""
Функции активации и их модульные обёртки.

Содержит две группы сущностей:

- чистые функции (``relu``, ``sigmoid``, ``tanh`` и их производные) —
  полезны для тестов и прямых вычислений;
- классы-модули (``ReLU``, ``Sigmoid``, ``Tanh``), наследующие
  :class:`Module` — используются в композиции слоёв, например
  внутри ``Sequential``.

Классы сохраняют выход последнего прямого прохода в поле ``y``
и используют его в ``backward`` — производные всех трёх активаций
выражаются через выход проще, чем через вход.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .core import Module

# ---------- Функции ----------

def relu(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Поэлементный ReLU: ``max(0, x)``.

    Args:
        x: Входной массив произвольной формы.

    Returns:
        Массив той же формы с отрицательными значениями, заменёнными на 0.
    """
    return np.where(x > 0, x, 0)


def relu_grad(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Производная ReLU по входу.

    Возвращает 1 там, где ``x > 0``, и 0 в остальных точках.
    В точке ``x = 0`` производная не определена; по соглашению
    используется 0.

    Args:
        x: Входной массив произвольной формы.

    Returns:
        Массив 0.0 и 1.0 той же формы, что ``x``.
    """
    return (x > 0).astype(np.float64)


def sigmoid(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Поэлементная сигмоида: ``1 / (1 + exp(-x))``.

    Args:
        x: Входной массив произвольной формы.

    Returns:
        Массив значений в интервале ``(0, 1)`` той же формы.
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_grad(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Производная сигмоиды по входу.

    Формула: ``s(x) * (1 - s(x))``, где ``s(x) = sigmoid(x)``.

    Args:
        x: Входной массив произвольной формы.

    Returns:
        Массив производных той же формы.
    """
    return sigmoid(x) * (1 - sigmoid(x))


def tanh(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Поэлементный гиперболический тангенс.

    Использует :func:`numpy.tanh`, который численно устойчив
    при больших ``|x|``.

    Args:
        x: Входной массив произвольной формы.

    Returns:
        Массив значений в интервале ``(-1, 1)`` той же формы.
    """
    return np.tanh(x)


def tanh_grad(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Производная tanh по входу.

    Формула: ``1 - tanh(x)**2``.

    Args:
        x: Входной массив произвольной формы.

    Returns:
        Массив производных той же формы.
    """
    return 1 - tanh(x) ** 2


# ---------- Классы-модули ----------

class ReLU(Module):
    """
    Модуль активации ReLU.

    Прямой проход: ``y = max(0, x)``. Обратный проход:
    ``dx = dout * (y > 0)``.

    Attributes:
        y: Сохранённый выход последнего ``forward``. ``None`` до первого
            вызова. Нужен для вычисления градиента в ``backward``.

    Example:
        >>> relu = ReLU()
        >>> y = relu(np.array([-1.0, 0.0, 2.0]))
        >>> y
        array([0., 0., 2.])
        >>> dy = relu.backward(np.array([1.0, 1.0, 1.0]))
        >>> dy
        array([0., 0., 1.])
    """

    def __init__(self) -> None:
        """Создаёт модуль без параметров."""
        super().__init__()
        self.y: NDArray[np.float64] | None = None

    def forward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Применяет ReLU поэлементно.

        Args:
            x: Входной массив произвольной формы.

        Returns:
            Массив той же формы с обнулёнными отрицательными значениями.
        """
        self.y = relu(x)
        return self.y

    def backward(self, dout: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Считает градиент по входу через маску ``y > 0``.

        Args:
            dout: Градиент по выходу, той же формы, что ``y``.

        Returns:
            Градиент по входу той же формы.

        Raises:
            AssertionError: Если :meth:`forward` не был вызван раньше.
        """
        assert self.y is not None, "forward must be called before backward"
        return dout * (self.y > 0)


class Sigmoid(Module):
    """
    Модуль активации Sigmoid.

    Прямой проход: ``y = 1 / (1 + exp(-x))``. Обратный проход:
    ``dx = dout * y * (1 - y)``.

    Attributes:
        y: Сохранённый выход последнего ``forward``. ``None`` до первого
            вызова.

    Example:
        >>> sigmoid = Sigmoid()
        >>> y = sigmoid(np.array([0.0]))
        >>> y
        array([0.5])
    """

    def __init__(self) -> None:
        """
        Создаёт модуль без параметров."""
        super().__init__()
        self.y: NDArray[np.float64] | None = None

    def forward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Применяет сигмоиду поэлементно.

        Args:
            x: Входной массив произвольной формы.

        Returns:
            Массив значений в ``(0, 1)`` той же формы.
        """
        self.y = sigmoid(x)
        return self.y

    def backward(self, dout: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Считает градиент по входу через сохранённый выход.

        Args:
            dout: Градиент по выходу, той же формы, что ``y``.

        Returns:
            Градиент по входу той же формы.

        Raises:
            AssertionError: Если :meth:`forward` не был вызван раньше.
        """
        assert self.y is not None, "forward must be called before backward"
        return dout * self.y * (1 - self.y)


class Tanh(Module):
    """
    Модуль активации Tanh.

    Прямой проход: ``y = tanh(x)``. Обратный проход:
    ``dx = dout * (1 - y**2)``.

    Attributes:
        y: Сохранённый выход последнего ``forward``. ``None`` до первого
            вызова.

    Example:
        >>> tanh = Tanh()
        >>> y = tanh(np.array([0.0]))
        >>> y
        array([0.])
    """

    def __init__(self) -> None:
        """Создаёт модуль без параметров."""
        super().__init__()
        self.y: NDArray[np.float64] | None = None

    def forward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Применяет tanh поэлементно.

        Args:
            x: Входной массив произвольной формы.

        Returns:
            Массив значений в ``(-1, 1)`` той же формы.
        """
        self.y = tanh(x)
        return self.y

    def backward(self, dout: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Считает градиент по входу через сохранённый выход.

        Args:
            dout: Градиент по выходу, той же формы, что ``y``.

        Returns:
            Градиент по входу той же формы.

        Raises:
            AssertionError: Если :meth:`forward` не был вызван раньше.
        """
        assert self.y is not None, "forward must be called before backward"
        return dout * (1 - self.y ** 2)
