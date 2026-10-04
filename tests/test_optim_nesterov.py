"""Тесты для оптимизатора Nesterov Accelerated Gradient (NAG)."""

from __future__ import annotations

import numpy as np
import pytest

from numpy_nn.nn.core import Parameter
from numpy_nn.optim import SGD, Momentum, Nesterov

# ---------- Конструктор и наследование ----------

def test_nesterov_is_momentum_subclass() -> None:
    """Nesterov наследуется от Momentum — значит, разделяет его состояние."""
    p = Parameter(np.array([1.0]))
    optimizer = Nesterov([p], lr=0.1)
    assert isinstance(optimizer, Momentum)


def test_init_stores_params_lr_beta() -> None:
    p = Parameter(np.array([1.0, 2.0]))
    optimizer = Nesterov([p], lr=0.1, beta=0.8)

    assert optimizer.params == [p]
    assert optimizer.lr == 0.1
    assert optimizer.beta == 0.8


def test_default_beta_is_0_9() -> None:
    p = Parameter(np.array([1.0]))
    optimizer = Nesterov([p], lr=0.1)
    assert optimizer.beta == 0.9


@pytest.mark.parametrize("beta", [-0.1, 1.0, 1.5])
def test_beta_out_of_range_raises(beta: float) -> None:
    """Валидация beta наследуется от Momentum."""
    p = Parameter(np.array([1.0]))
    with pytest.raises(ValueError):
        Nesterov([p], lr=0.1, beta=beta)


def test_buffers_initialized_to_zeros() -> None:
    """Буферы скорости инициализируются как в Momentum."""
    p = Parameter(np.array([1.0, 2.0, 3.0]))
    optimizer = Nesterov([p], lr=0.1)

    assert len(optimizer.v) == 1
    np.testing.assert_array_equal(optimizer.v[0], np.zeros(3))


# ---------- Формула первого шага ----------

def test_first_step_velocity_equals_grad() -> None:
    """После первого шага v_1 = grad (как в Momentum)."""
    p = Parameter(np.array([1.0, 2.0]))
    p.grad = np.array([0.5, -0.5])
    optimizer = Nesterov([p], lr=0.1, beta=0.9)

    optimizer.step()

    np.testing.assert_allclose(optimizer.v[0], np.array([0.5, -0.5]))


def test_first_step_param_update() -> None:
    """p_1 = p_0 - lr * (1 + beta) * grad.

    Эффективный градиент на первом шаге: d = grad + beta * v_1 = (1 + beta) * grad.
    """
    p = Parameter(np.array([1.0]))
    p.grad = np.array([1.0])
    optimizer = Nesterov([p], lr=0.1, beta=0.9)

    optimizer.step()

    # p1 = 1.0 - 0.1 * (1 + 0.9) * 1.0 = 1.0 - 0.19 = 0.81
    np.testing.assert_allclose(p.data, np.array([0.81]))


def test_first_step_larger_than_momentum() -> None:
    """На первом шаге NAG делает шаг больше, чем Momentum при тех же условиях."""
    init = np.array([1.0])

    p_nag = Parameter(init.copy())
    p_nag.grad = np.array([1.0])
    p_mom = Parameter(init.copy())
    p_mom.grad = np.array([1.0])

    Nesterov([p_nag], lr=0.1, beta=0.9).step()
    Momentum([p_mom], lr=0.1, beta=0.9).step()

    # NAG ушёл дальше (значение меньше), потому что d = (1 + beta) * grad
    assert p_nag.data[0] < p_mom.data[0]


# ---------- Формула второго шага ----------

def test_two_steps_manual_formula() -> None:
    """Полный ручной расчёт двух шагов NAG."""
    p = Parameter(np.array([1.0]))
    p.grad = np.array([1.0])
    optimizer = Nesterov([p], lr=0.1, beta=0.9)

    optimizer.step()
    # v1 = 1.0, d1 = 1.0 + 0.9 * 1.0 = 1.9, p1 = 1.0 - 0.1 * 1.9 = 0.81

    p.grad = np.array([1.0])
    optimizer.step()
    # v2 = 0.9 * 1.0 + 1.0 = 1.9
    # d2 = 1.0 + 0.9 * 1.9 = 1.0 + 1.71 = 2.71
    # p2 = 0.81 - 0.1 * 2.71 = 0.539

    np.testing.assert_allclose(optimizer.v[0], np.array([1.9]))
    np.testing.assert_allclose(p.data, np.array([0.539]))


def test_d_uses_updated_v() -> None:
    """Ключевая проверка: d использует уже обновлённый v, а не старый.

    Если бы d считался по старому v (v_old), на втором шаге:
      d2_old = grad + beta * v_old = 1.0 + 0.9 * 1.0 = 1.9
      p2_old = 0.81 - 0.1 * 1.9 = 0.62

    Но по формуле PyTorch d2 = 2.71, и p2 = 0.539.
    Проверяем, что результат ближе к 0.539, чем к 0.62.
    """
    p = Parameter(np.array([1.0]))
    p.grad = np.array([1.0])
    optimizer = Nesterov([p], lr=0.1, beta=0.9)

    optimizer.step()
    p.grad = np.array([1.0])
    optimizer.step()

    # ожидаемое значение именно 0.539, не 0.62
    assert abs(p.data[0] - 0.539) < 1e-10


# ---------- Отличие от Momentum ----------

def test_differs_from_momentum_on_same_grads() -> None:
    """При одинаковых начальных условиях и градиентах NAG и Momentum
    дают разные траектории."""
    rng = np.random.default_rng(0)
    init = rng.standard_normal(5)

    p_nag = Parameter(init.copy())
    p_mom = Parameter(init.copy())
    opt_nag = Nesterov([p_nag], lr=0.1, beta=0.9)
    opt_mom = Momentum([p_mom], lr=0.1, beta=0.9)

    for _ in range(5):
        grad = rng.standard_normal(5)
        p_nag.grad = grad.copy()
        p_mom.grad = grad.copy()
        opt_nag.step()
        opt_mom.step()

    assert not np.allclose(p_nag.data, p_mom.data)


# ---------- beta = 0 ----------

def test_beta_zero_behaves_like_sgd() -> None:
    """При beta = 0: v = grad, d = grad, обновление как в SGD."""
    rng = np.random.default_rng(1)
    init = rng.standard_normal(5)

    p_nag = Parameter(init.copy())
    p_sgd = Parameter(init.copy())
    opt_nag = Nesterov([p_nag], lr=0.1, beta=0.0)
    opt_sgd = SGD([p_sgd], lr=0.1)

    for _ in range(5):
        grad = rng.standard_normal(5)
        p_nag.grad = grad.copy()
        p_sgd.grad = grad.copy()
        opt_nag.step()
        opt_sgd.step()

    np.testing.assert_allclose(p_nag.data, p_sgd.data)


# ---------- None-градиенты ----------

def test_step_skips_none_grad() -> None:
    p = Parameter(np.array([1.0, 2.0]))
    optimizer = Nesterov([p], lr=0.5)

    optimizer.step()

    np.testing.assert_array_equal(p.data, np.array([1.0, 2.0]))
    np.testing.assert_array_equal(optimizer.v[0], np.array([0.0, 0.0]))


def test_step_mixed_none_and_real_grad() -> None:
    p_no_grad = Parameter(np.array([1.0]))
    p_with_grad = Parameter(np.array([1.0]))
    p_with_grad.grad = np.array([1.0])

    optimizer = Nesterov([p_no_grad, p_with_grad], lr=0.1, beta=0.9)
    optimizer.step()

    np.testing.assert_array_equal(p_no_grad.data, np.array([1.0]))
    # для p_with_grad: d = 1.0 + 0.9 * 1.0 = 1.9, p = 1.0 - 0.1 * 1.9 = 0.81
    np.testing.assert_allclose(p_with_grad.data, np.array([0.81]))


# ---------- zero_grad (наследуется) ----------

def test_zero_grad_resets_grads_but_not_buffers() -> None:
    p = Parameter(np.array([1.0]))
    p.grad = np.array([0.5])
    optimizer = Nesterov([p], lr=0.1)

    optimizer.step()
    v_before = optimizer.v[0].copy()

    optimizer.zero_grad()

    assert p.grad is None
    np.testing.assert_array_equal(optimizer.v[0], v_before)


# ---------- In-place ----------

def test_param_data_updated_in_place() -> None:
    p = Parameter(np.array([1.0, 2.0]))
    p.grad = np.array([0.1, 0.2])
    data_before = p.data
    optimizer = Nesterov([p], lr=0.1)

    optimizer.step()

    assert p.data is data_before


def test_velocity_updated_in_place() -> None:
    p = Parameter(np.array([1.0, 2.0]))
    p.grad = np.array([0.1, 0.2])
    optimizer = Nesterov([p], lr=0.1)
    v_before = optimizer.v[0]

    optimizer.step()

    assert optimizer.v[0] is v_before


# ---------- Сходимость ----------

def test_converges_on_quadratic() -> None:
    p = Parameter(np.array([1.0, -2.0, 3.0]))
    optimizer = Nesterov([p], lr=0.05, beta=0.9)

    for _ in range(200):
        p.grad = p.data.copy()
        optimizer.step()

    assert np.allclose(p.data, 0.0, atol=1e-4)


# ---------- Сквозной тест с моделью ----------

def test_multiple_steps_decrease_loss() -> None:
    from numpy_nn.nn import Linear, MSELoss, Sequential

    rng = np.random.default_rng(0)
    model = Sequential(Linear(4, 1))
    criterion = MSELoss()
    optimizer = Nesterov(model.parameters(), lr=0.05, beta=0.9)

    x = rng.standard_normal((20, 4))
    y = rng.standard_normal((20, 1))

    losses = []
    for _ in range(30):
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        losses.append(loss)
        dpred = criterion.backward()
        model.backward(dpred)
        optimizer.step()

    assert losses[-1] < losses[0]
