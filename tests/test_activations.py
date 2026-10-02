"""Тесты для функций активации и классов-модулей ReLU, Sigmoid, Tanh."""

from __future__ import annotations

import numpy as np
import pytest

from numpy_nn.nn.activations import (
    ReLU,
    Sigmoid,
    Tanh,
    relu,
    relu_grad,
    sigmoid,
    sigmoid_grad,
    tanh,
    tanh_grad,
)

# ---------- Функции: ReLU ----------

def test_relu_values() -> None:
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    expected = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
    np.testing.assert_array_equal(relu(x), expected)


def test_relu_shape_preserved() -> None:
    x = np.random.randn(3, 4)
    assert relu(x).shape == (3, 4)


def test_relu_grad_values() -> None:
    x = np.array([-1.0, 0.0, 1.0])
    expected = np.array([0.0, 0.0, 1.0])
    np.testing.assert_array_equal(relu_grad(x), expected)


def test_relu_grad_dtype_is_float64() -> None:
    x = np.array([-1.0, 1.0])
    assert relu_grad(x).dtype == np.float64


# ---------- Функции: Sigmoid ----------

def test_sigmoid_at_zero() -> None:
    assert sigmoid(np.array([0.0]))[0] == pytest.approx(0.5)


def test_sigmoid_range() -> None:
    """Все значения лежат строго в (0, 1)."""
    x = np.linspace(-10, 10, 100)
    y = sigmoid(x)
    assert np.all(y > 0)
    assert np.all(y < 1)


def test_sigmoid_symmetry() -> None:
    """sigmoid(-x) = 1 - sigmoid(x)."""
    x = np.linspace(-5, 5, 50)
    np.testing.assert_allclose(sigmoid(-x), 1 - sigmoid(x), atol=1e-12)


def test_sigmoid_grad_at_zero() -> None:
    """sigmoid'(0) = 0.25."""
    assert sigmoid_grad(np.array([0.0]))[0] == pytest.approx(0.25)


def test_sigmoid_grad_values() -> None:
    """Проверка через ручной расчёт на нескольких точках."""
    x = np.array([-1.0, 0.0, 1.0])
    s = sigmoid(x)
    expected = s * (1 - s)
    np.testing.assert_allclose(sigmoid_grad(x), expected)


# ---------- Функции: Tanh ----------

def test_tanh_at_zero() -> None:
    assert tanh(np.array([0.0]))[0] == pytest.approx(0.0)


def test_tanh_range() -> None:
    """Все значения лежат строго в (-1, 1)."""
    x = np.linspace(-10, 10, 100)
    y = tanh(x)
    assert np.all(y > -1)
    assert np.all(y < 1)


def test_tanh_odd() -> None:
    """tanh(-x) = -tanh(x)."""
    x = np.linspace(-3, 3, 30)
    np.testing.assert_allclose(tanh(-x), -tanh(x), atol=1e-12)


def test_tanh_numerical_stability() -> None:
    """На больших |x| tanh не должен давать nan или inf."""
    x = np.array([-1000.0, -100.0, 100.0, 1000.0])
    y = tanh(x)
    assert np.all(np.isfinite(y))
    np.testing.assert_allclose(y, np.array([-1.0, -1.0, 1.0, 1.0]), atol=1e-6)


def test_tanh_grad_at_zero() -> None:
    """tanh'(0) = 1."""
    assert tanh_grad(np.array([0.0]))[0] == pytest.approx(1.0)


def test_tanh_grad_values() -> None:
    x = np.array([-1.0, 0.0, 1.0])
    expected = 1 - np.tanh(x) ** 2
    np.testing.assert_allclose(tanh_grad(x), expected)


# ---------- Классы: общие проверки ----------

@pytest.mark.parametrize("cls", [ReLU, Sigmoid, Tanh])
def test_module_forward_shape(cls: type) -> None:
    layer = cls()
    x = np.random.randn(4, 5)
    y = layer(x)
    assert y.shape == (4, 5)


@pytest.mark.parametrize("cls", [ReLU, Sigmoid, Tanh])
def test_module_saves_y(cls: type) -> None:
    layer = cls()
    x = np.random.randn(3, 3)
    y = layer(x)
    assert layer.y is not None
    np.testing.assert_array_equal(layer.y, y)


@pytest.mark.parametrize("cls", [ReLU, Sigmoid, Tanh])
def test_module_backward_requires_forward(cls: type) -> None:
    layer = cls()
    dout = np.random.randn(3, 3)
    with pytest.raises(AssertionError):
        layer.backward(dout)


@pytest.mark.parametrize("cls", [ReLU, Sigmoid, Tanh])
def test_module_backward_shape(cls: type) -> None:
    layer = cls()
    x = np.random.randn(4, 5)
    layer(x)
    dout = np.random.randn(4, 5)
    dx = layer.backward(dout)
    assert dx.shape == (4, 5)


@pytest.mark.parametrize("cls", [ReLU, Sigmoid, Tanh])
def test_module_has_no_parameters(cls: type) -> None:
    layer = cls()
    assert layer.parameters() == []


# ---------- Классы: значения ----------

def test_relu_module_values() -> None:
    layer = ReLU()
    y = layer(np.array([-1.0, 0.0, 1.0]))
    np.testing.assert_array_equal(y, np.array([0.0, 0.0, 1.0]))


def test_sigmoid_module_values() -> None:
    layer = Sigmoid()
    y = layer(np.array([0.0]))
    assert y[0] == pytest.approx(0.5)


def test_tanh_module_values() -> None:
    layer = Tanh()
    y = layer(np.array([0.0]))
    assert y[0] == pytest.approx(0.0)


# ---------- Gradient checking ----------

def _numerical_grad(
    layer,
    x: np.ndarray,
    dout: np.ndarray,
    param_index: tuple[int, ...],
    eps: float = 1e-6,
) -> float:
    """Численная производная скалярной функции L(x) = <layer(x), dout>
    по элементу x[param_index].
    """
    x_plus = x.copy()
    x_plus[param_index] += eps
    x_minus = x.copy()
    x_minus[param_index] -= eps

    l_plus = float((layer(x_plus) * dout).sum())
    l_minus = float((layer(x_minus) * dout).sum())
    return (l_plus - l_minus) / (2 * eps)


def _check_gradient(layer, x: np.ndarray, dout: np.ndarray) -> None:
    """Проверяет, что backward даёт градиент, совпадающий
    с численной производной в каждой точке.

    Идея: L(x) = <layer(x), dout> — скалярная функция от x.
    Её градиент по x — это в точности то, что возвращает backward,
    потому что dL/dy = dout, а дальше идёт цепное правило слоя.
    """
    # прогоняем forward один раз, чтобы layer.y сохранился
    layer(x)
    dx_analytic = layer.backward(dout)

    # сбрасываем состояние, чтобы forward внутри численного дифференцирования
    # не портил сохранённый y
    for idx in np.ndindex(x.shape):
        num = _numerical_grad(layer, x, dout, idx)
        assert abs(num - dx_analytic[idx]) < 1e-6, (
            f"gradient mismatch at {idx}: analytic={dx_analytic[idx]}, "
            f"numeric={num}"
        )


def test_relu_gradient_check() -> None:
    rng = np.random.default_rng(0)
    layer = ReLU()
    # избегаем значений ровно у нуля — там производная разрывна
    x = rng.standard_normal((2, 3)) + 0.1
    dout = rng.standard_normal((2, 3))
    _check_gradient(layer, x, dout)


def test_sigmoid_gradient_check() -> None:
    rng = np.random.default_rng(1)
    layer = Sigmoid()
    x = rng.standard_normal((2, 3))
    dout = rng.standard_normal((2, 3))
    _check_gradient(layer, x, dout)


def test_tanh_gradient_check() -> None:
    rng = np.random.default_rng(2)
    layer = Tanh()
    x = rng.standard_normal((2, 3))
    dout = rng.standard_normal((2, 3))
    _check_gradient(layer, x, dout)
