"""Тесты для контейнера Sequential."""

from __future__ import annotations

import numpy as np
import pytest

from numpy_nn.nn.activations import ReLU, Sigmoid, Tanh
from numpy_nn.nn.layers import Linear
from numpy_nn.nn.sequential import Sequential

# ---------- Инициализация и протокол последовательности ----------

def test_empty_sequential_len() -> None:
    model = Sequential()
    assert len(model) == 0


def test_len_matches_number_of_layers() -> None:
    model = Sequential(Linear(4, 3), ReLU(), Linear(3, 2))
    assert len(model) == 3


def test_getitem_returns_layers_in_order() -> None:
    l1 = Linear(4, 3)
    l2 = ReLU()
    l3 = Linear(3, 2)
    model = Sequential(l1, l2, l3)

    assert model[0] is l1
    assert model[1] is l2
    assert model[2] is l3


def test_getitem_supports_negative_index() -> None:
    l1 = Linear(4, 3)
    l2 = ReLU()
    model = Sequential(l1, l2)

    assert model[-1] is l2
    assert model[-2] is l1


# ---------- Forward ----------

def test_empty_sequential_is_identity() -> None:
    """Пустой Sequential возвращает вход без изменений."""
    model = Sequential()
    x = np.random.randn(5, 4)
    y = model(x)
    np.testing.assert_array_equal(y, x)


def test_forward_shape_chain() -> None:
    model = Sequential(
        Linear(4, 8),
        ReLU(),
        Linear(8, 2),
    )
    x = np.random.randn(5, 4)
    y = model(x)
    assert y.shape == (5, 2)


def test_forward_passes_through_activation() -> None:
    """Проверяем, что выход активации реально используется дальше."""
    model = Sequential(
        Linear(4, 3),
        ReLU(),
    )
    x = np.random.randn(5, 4)
    y = model(x)
    # после ReLU все значения неотрицательные
    assert np.all(y >= 0)


def test_forward_equivalent_to_manual_chain() -> None:
    """Сравниваем Sequential с ручным применением тех же слоёв."""
    rng = np.random.default_rng(0)
    l1 = Linear(4, 5)
    act = Tanh()
    l2 = Linear(5, 3)

    model = Sequential(l1, act, l2)
    x = rng.standard_normal((6, 4))

    # ручной прогон
    h1 = l1(x)
    h2 = act(h1)
    y_manual = l2(h2)

    y_seq = model(x)
    np.testing.assert_allclose(y_seq, y_manual)


# ---------- Backward ----------

def test_backward_shape_chain() -> None:
    model = Sequential(
        Linear(4, 8),
        ReLU(),
        Linear(8, 2),
    )
    x = np.random.randn(5, 4)
    model(x)
    dout = np.random.randn(5, 2)
    dx = model.backward(dout)
    assert dx.shape == (5, 4)


def test_backward_empty_sequential() -> None:
    """Пустой Sequential возвращает dout без изменений."""
    model = Sequential()
    dout = np.random.randn(3, 2)
    dx = model.backward(dout)
    np.testing.assert_array_equal(dx, dout)


def test_backward_writes_grads_to_parameters() -> None:
    """После backward у всех параметров есть градиенты."""
    model = Sequential(
        Linear(4, 8),
        ReLU(),
        Linear(8, 2),
    )
    x = np.random.randn(5, 4)
    model(x)
    dout = np.random.randn(5, 2)
    model.backward(dout)

    for p in model.parameters():
        assert p.grad is not None
        assert p.grad.shape == p.data.shape


# ---------- Параметры ----------

def test_parameters_empty_for_stateless_layers() -> None:
    model = Sequential(ReLU(), Sigmoid(), Tanh())
    assert model.parameters() == []


def test_parameters_flat_list() -> None:
    """parameters() возвращает плоский список, не список списков."""
    model = Sequential(
        Linear(4, 3),
        ReLU(),
        Linear(3, 2),
    )
    params = model.parameters()

    # два Linear, у каждого weight + bias → 4 параметра
    assert len(params) == 4
    # все элементы — Parameter, не list
    for p in params:
        assert hasattr(p, "data")
        assert hasattr(p, "grad")


def test_parameters_order() -> None:
    """Порядок: сначала параметры первого слоя, потом второго."""
    l1 = Linear(4, 3)
    l2 = Linear(3, 2)
    model = Sequential(l1, ReLU(), l2)

    params = model.parameters()
    assert params[0] is l1.weight
    assert params[1] is l1.bias
    assert params[2] is l2.weight
    assert params[3] is l2.bias


def test_zero_grad_resets_all_parameters() -> None:
    """zero_grad через наследование от Module обходит все параметры."""
    model = Sequential(
        Linear(4, 3),
        ReLU(),
        Linear(3, 2),
    )
    x = np.random.randn(5, 4)
    model(x)
    dout = np.random.randn(5, 2)
    model.backward(dout)

    for p in model.parameters():
        assert p.grad is not None

    model.zero_grad()

    for p in model.parameters():
        assert p.grad is None


# ---------- Вложенные Sequential ----------

def test_nested_sequential_forward() -> None:
    """Sequential внутри Sequential работает как обычный модуль."""
    inner = Sequential(Linear(4, 5), ReLU())
    outer = Sequential(inner, Linear(5, 2))

    x = np.random.randn(3, 4)
    y = outer(x)
    assert y.shape == (3, 2)


def test_nested_sequential_parameters() -> None:
    """Параметры вложенного Sequential собираются рекурсивно."""
    inner = Sequential(Linear(4, 5), ReLU())
    outer = Sequential(inner, Linear(5, 2))

    params = outer.parameters()
    # inner: Linear(4,5) → 2 параметра, outer: Linear(5,2) → 2 параметра
    assert len(params) == 4


def test_nested_sequential_backward() -> None:
    inner = Sequential(Linear(4, 5), ReLU())
    outer = Sequential(inner, Linear(5, 2))

    x = np.random.randn(3, 4)
    outer(x)
    dout = np.random.randn(3, 2)
    dx = outer.backward(dout)
    assert dx.shape == (3, 4)


# ---------- Gradient checking ----------

def _numerical_grad_param(
    model: Sequential,
    param,
    x: np.ndarray,
    dout: np.ndarray,
    param_index: tuple[int, ...],
    eps: float = 1e-6,
) -> float:
    """Численная производная L = <model(x), dout> по одному элементу param.data."""
    original = param.data[param_index]

    param.data[param_index] = original + eps
    l_plus = float((model(x) * dout).sum())

    param.data[param_index] = original - eps
    l_minus = float((model(x) * dout).sum())

    param.data[param_index] = original
    return (l_plus - l_minus) / (2 * eps)


def _check_model_gradient(model: Sequential, x: np.ndarray, dout: np.ndarray) -> None:
    """Проверяет все параметры модели через численный градиент."""
    model(x)
    model.backward(dout)

    for param in model.parameters():
        assert param.grad is not None
        for idx in np.ndindex(param.data.shape):
            num = _numerical_grad_param(model, param, x, dout, idx)
            analytic = param.grad[idx]
            assert abs(num - analytic) < 1e-5, (
                f"mismatch at {idx}: analytic={analytic}, numeric={num}"
            )


def test_gradient_check_linear_relu_linear() -> None:
    """Сквозная проверка градиентов на цепочке Linear → ReLU → Linear."""
    rng = np.random.default_rng(42)
    model = Sequential(
        Linear(4, 5),
        ReLU(),
        Linear(5, 3),
    )
    # сдвигаем вход, чтобы активации не сидели в нуле
    x = rng.standard_normal((2, 4)) + 0.5
    dout = rng.standard_normal((2, 3))
    _check_model_gradient(model, x, dout)


def test_gradient_check_linear_tanh_linear() -> None:
    """Проверка с Tanh — гладкая активация, точность выше."""
    rng = np.random.default_rng(43)
    model = Sequential(
        Linear(3, 4),
        Tanh(),
        Linear(4, 2),
    )
    x = rng.standard_normal((2, 3))
    dout = rng.standard_normal((2, 2))
    _check_model_gradient(model, x, dout)


def test_gradient_check_nested() -> None:
    """Проверка на вложенном Sequential."""
    rng = np.random.default_rng(44)
    inner = Sequential(Linear(3, 4), Sigmoid())
    model = Sequential(inner, Linear(4, 2))

    x = rng.standard_normal((2, 3))
    dout = rng.standard_normal((2, 2))
    _check_model_gradient(model, x, dout)
