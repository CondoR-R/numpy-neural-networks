"""Тесты для полносвязного слоя Linear."""

from __future__ import annotations

import numpy as np
import pytest

from numpy_nn.nn.core import Parameter
from numpy_nn.nn.layers import Linear

# ---------- Инициализация ----------

def test_shapes_after_init() -> None:
    layer = Linear(in_features=4, out_features=3)
    assert layer.weight.data.shape == (3, 4)
    assert layer.bias.data.shape == (3,)
    assert layer.weight.grad is None
    assert layer.bias.grad is None


def test_bias_initialized_to_zero() -> None:
    layer = Linear(in_features=10, out_features=5)
    np.testing.assert_array_equal(layer.bias.data, np.zeros(5))


def test_xavier_uniform_bounds() -> None:
    """Все веса лежат в [-limit, +limit]."""
    in_features, out_features = 100, 50
    limit = np.sqrt(6 / (in_features + out_features))
    layer = Linear(in_features=in_features, out_features=out_features)
    assert np.all(layer.weight.data >= -limit)
    assert np.all(layer.weight.data <= limit)


def test_xavier_uniform_variance() -> None:
    """Эмпирическая дисперсия близка к теоретической limit^2 / 3."""
    in_features, out_features = 1000, 1000
    limit = np.sqrt(6 / (in_features + out_features))
    expected_var = limit ** 2 / 3

    # усредняем по нескольким инициализациям, чтобы уменьшить случайный шум
    vars_ = []
    for _ in range(20):
        layer = Linear(in_features=in_features, out_features=out_features)
        vars_.append(layer.weight.data.var())
    mean_var = float(np.mean(vars_))

    # допуск 5% — на 20 прогонах по 10^6 значений стабильно попадает
    assert abs(mean_var - expected_var) / expected_var < 0.05


# ---------- Forward ----------

def test_forward_shape_batch() -> None:
    layer = Linear(in_features=4, out_features=3)
    x = np.random.randn(5, 4)
    y = layer(x)
    assert y.shape == (5, 3)


def test_forward_shape_single() -> None:
    layer = Linear(in_features=4, out_features=3)
    x = np.random.randn(4)
    y = layer(x)
    assert y.shape == (3,)


def test_forward_saves_x() -> None:
    layer = Linear(in_features=4, out_features=3)
    x = np.random.randn(5, 4)
    layer(x)
    assert layer.x is not None
    np.testing.assert_array_equal(layer.x, x)


def test_forward_formula() -> None:
    """Проверяем, что forward совпадает с ручной формулой X @ W.T + b."""
    layer = Linear(in_features=3, out_features=2)
    # зафиксируем веса и смещения, чтобы результат был детерминирован
    layer.weight.data = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]])
    layer.bias.data = np.array([0.5, -0.5])

    x = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    expected = x @ layer.weight.data.T + layer.bias.data
    np.testing.assert_allclose(layer(x), expected)


# ---------- Backward ----------

def test_backward_shapes() -> None:
    layer = Linear(in_features=4, out_features=3)
    x = np.random.randn(5, 4)
    layer(x)
    dout = np.random.randn(5, 3)
    dx = layer.backward(dout)

    assert layer.weight.grad is not None
    assert layer.bias.grad is not None
    assert layer.weight.grad.shape == (3, 4)
    assert layer.bias.grad.shape == (3,)
    assert dx.shape == (5, 4)


def test_backward_requires_forward() -> None:
    layer = Linear(in_features=4, out_features=3)
    dout = np.random.randn(5, 3)
    with pytest.raises(AssertionError):
        layer.backward(dout)


def test_backward_formula() -> None:
    """Проверяем, что backward совпадает с ручными формулами."""
    layer = Linear(in_features=3, out_features=2)
    layer.weight.data = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]])
    layer.bias.data = np.array([0.5, -0.5])

    x = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    layer(x)

    dout = np.array([[1.0, 0.0], [0.0, 1.0]])
    dx = layer.backward(dout)

    expected_dw = dout.T @ x
    expected_db = dout.sum(axis=0)
    expected_dx = dout @ layer.weight.data

    np.testing.assert_allclose(layer.weight.grad, expected_dw)
    np.testing.assert_allclose(layer.bias.grad, expected_db)
    np.testing.assert_allclose(dx, expected_dx)


def test_backward_zero_grad() -> None:
    """Суммарный градиент от нулевого dout должен быть нулевым."""
    layer = Linear(in_features=4, out_features=3)
    x = np.random.randn(5, 4)
    layer(x)
    dout = np.zeros((5, 3))
    dx = layer.backward(dout)

    np.testing.assert_array_equal(layer.weight.grad, np.zeros((3, 4)))
    np.testing.assert_array_equal(layer.bias.grad, np.zeros(3))
    np.testing.assert_array_equal(dx, np.zeros((5, 4)))


# ---------- Параметры и zero_grad ----------

def test_parameters_returns_weight_and_bias() -> None:
    layer = Linear(in_features=4, out_features=3)
    params = layer.parameters()
    assert len(params) == 2
    assert params[0] is layer.weight
    assert params[1] is layer.bias


def test_zero_grad_resets_to_none() -> None:
    layer = Linear(in_features=4, out_features=3)
    x = np.random.randn(5, 4)
    layer(x)
    dout = np.random.randn(5, 3)
    layer.backward(dout)

    assert layer.weight.grad is not None
    assert layer.bias.grad is not None

    layer.zero_grad()

    assert layer.weight.grad is None
    assert layer.bias.grad is None


# ---------- Gradient checking ----------

def _numerical_grad(
    layer: Linear,
    param: Parameter,
    x: np.ndarray,
    loss_grad: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Считает численный градиент параметра через конечные разности.

    loss_grad — градиент loss по выходу слоя (dout), фиксированный.
    Аппроксимируем dL/dp[i] = (L(p+eps) - L(p-eps)) / (2*eps),
    где L(p) = <layer(x), loss_grad> — скалярное произведение выхода
    на фиксированный dout, что эквивалентно взятию производной от
    линейной функции потерь в точке выхода.
    """
    num_grad = np.zeros_like(param.data)
    it = np.nditer(param.data, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        original = param.data[idx]

        param.data[idx] = original + eps
        y_plus = layer(x)
        loss_plus = float((y_plus * loss_grad).sum())

        param.data[idx] = original - eps
        y_minus = layer(x)
        loss_minus = float((y_minus * loss_grad).sum())

        param.data[idx] = original
        num_grad[idx] = (loss_plus - loss_minus) / (2 * eps)
        it.iternext()
    return num_grad


def test_gradient_check_weight() -> None:
    """Численный градиент по весам совпадает с аналитическим."""
    rng = np.random.default_rng(0)
    layer = Linear(in_features=4, out_features=3)
    layer.weight.data = rng.standard_normal((3, 4))
    layer.bias.data = rng.standard_normal(3)

    x = rng.standard_normal((5, 4))
    layer(x)
    dout = rng.standard_normal((5, 3))

    layer.backward(dout)
    assert layer.weight.grad is not None
    analytic = layer.weight.grad.copy()

    # переинициализируем слой, чтобы backward не запускался заново
    layer2 = Linear(in_features=4, out_features=3)
    layer2.weight.data = layer.weight.data.copy()
    layer2.bias.data = layer.bias.data.copy()

    numeric = _numerical_grad(layer2, layer2.weight, x, dout)

    np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-7)


def test_gradient_check_bias() -> None:
    """Численный градиент по смещениям совпадает с аналитическим."""
    rng = np.random.default_rng(1)
    layer = Linear(in_features=4, out_features=3)
    layer.weight.data = rng.standard_normal((3, 4))
    layer.bias.data = rng.standard_normal(3)

    x = rng.standard_normal((5, 4))
    layer(x)
    dout = rng.standard_normal((5, 3))

    layer.backward(dout)
    assert layer.bias.grad is not None
    analytic = layer.bias.grad.copy()

    layer2 = Linear(in_features=4, out_features=3)
    layer2.weight.data = layer.weight.data.copy()
    layer2.bias.data = layer.bias.data.copy()

    numeric = _numerical_grad(layer2, layer2.bias, x, dout)

    np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-7)
