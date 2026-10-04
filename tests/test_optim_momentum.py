"""Тесты для оптимизатора Momentum."""

from __future__ import annotations

import numpy as np
import pytest

from numpy_nn.nn.core import Parameter
from numpy_nn.optim import SGD, Momentum

# ---------- Конструктор ----------

def test_init_stores_params_lr_beta() -> None:
    p = Parameter(np.array([1.0, 2.0]))
    optimizer = Momentum([p], lr=0.1, beta=0.8)

    assert optimizer.params == [p]
    assert optimizer.lr == 0.1
    assert optimizer.beta == 0.8


def test_default_beta_is_0_9() -> None:
    p = Parameter(np.array([1.0]))
    optimizer = Momentum([p], lr=0.1)
    assert optimizer.beta == 0.9


@pytest.mark.parametrize("beta", [-0.1, -1.0, 1.0, 1.5, 2.0])
def test_beta_out_of_range_raises(beta: float) -> None:
    p = Parameter(np.array([1.0]))
    with pytest.raises(ValueError):
        Momentum([p], lr=0.1, beta=beta)


def test_beta_zero_allowed() -> None:
    p = Parameter(np.array([1.0]))
    optimizer = Momentum([p], lr=0.1, beta=0.0)
    assert optimizer.beta == 0.0


def test_beta_close_to_one_allowed() -> None:
    p = Parameter(np.array([1.0]))
    optimizer = Momentum([p], lr=0.1, beta=0.999)
    assert optimizer.beta == 0.999


def test_buffers_initialized_to_zeros() -> None:
    p1 = Parameter(np.array([1.0, 2.0, 3.0]))
    p2 = Parameter(np.zeros((2, 4)))
    optimizer = Momentum([p1, p2], lr=0.1)

    assert len(optimizer.v) == 2
    np.testing.assert_array_equal(optimizer.v[0], np.zeros(3))
    np.testing.assert_array_equal(optimizer.v[1], np.zeros((2, 4)))


def test_buffers_shape_and_dtype_match_params() -> None:
    p1 = Parameter(np.array([1.0, 2.0, 3.0]))
    p2 = Parameter(np.zeros((2, 4)))
    optimizer = Momentum([p1, p2], lr=0.1)

    assert optimizer.v[0].shape == p1.data.shape
    assert optimizer.v[0].dtype == np.float64
    assert optimizer.v[1].shape == p2.data.shape
    assert optimizer.v[1].dtype == np.float64


def test_init_copies_params_list() -> None:
    """Изменение внешнего списка не влияет на оптимизатор."""
    p1 = Parameter(np.array([1.0]))
    p2 = Parameter(np.array([2.0]))
    original = [p1]
    optimizer = Momentum(original, lr=0.1)

    original.append(p2)

    assert len(optimizer.params) == 1
    assert len(optimizer.v) == 1


# ---------- Один шаг ----------

def test_first_step_velocity_equals_grad() -> None:
    """При первом шаге v_1 = 0 * beta + grad = grad."""
    p = Parameter(np.array([1.0, 2.0]))
    p.grad = np.array([0.5, -0.5])
    optimizer = Momentum([p], lr=0.1, beta=0.9)

    optimizer.step()

    np.testing.assert_allclose(optimizer.v[0], np.array([0.5, -0.5]))


def test_first_step_param_update() -> None:
    """p_1 = p_0 - lr * v_1 = p_0 - lr * grad."""
    p = Parameter(np.array([1.0, 2.0]))
    p.grad = np.array([0.5, -0.5])
    optimizer = Momentum([p], lr=0.1, beta=0.9)

    optimizer.step()

    expected = np.array([1.0 - 0.05, 2.0 + 0.05])
    np.testing.assert_allclose(p.data, expected)


def test_step_direction() -> None:
    """Положительный градиент уменьшает параметр, отрицательный — увеличивает."""
    p_pos = Parameter(np.array([5.0]))
    p_pos.grad = np.array([1.0])
    p_neg = Parameter(np.array([5.0]))
    p_neg.grad = np.array([-1.0])

    optimizer = Momentum([p_pos, p_neg], lr=0.1)
    optimizer.step()

    assert p_pos.data[0] < 5.0
    assert p_neg.data[0] > 5.0


def test_step_updates_multiple_params() -> None:
    p1 = Parameter(np.array([1.0]))
    p1.grad = np.array([0.1])
    p2 = Parameter(np.array([2.0]))
    p2.grad = np.array([0.2])
    optimizer = Momentum([p1, p2], lr=0.5)

    optimizer.step()

    np.testing.assert_allclose(p1.data, np.array([0.95]))
    np.testing.assert_allclose(p2.data, np.array([1.9]))


# ---------- Накопление момента ----------

def test_velocity_accumulates_over_steps() -> None:
    """v_2 = beta * v_1 + g_2 = beta * g_1 + g_2 при одинаковых градиентах."""
    p = Parameter(np.array([0.0]))
    optimizer = Momentum([p], lr=0.1, beta=0.9)

    p.grad = np.array([1.0])
    optimizer.step()
    v1 = optimizer.v[0].copy()

    p.grad = np.array([1.0])
    optimizer.step()
    v2 = optimizer.v[0].copy()

    np.testing.assert_allclose(v1, np.array([1.0]))
    np.testing.assert_allclose(v2, np.array([0.9 * 1.0 + 1.0]))


def test_two_steps_manual_formula() -> None:
    """Полный ручной расчёт двух шагов."""
    p = Parameter(np.array([1.0]))
    p.grad = np.array([1.0])
    optimizer = Momentum([p], lr=0.1, beta=0.9)

    optimizer.step()
    # v1 = 1.0, p1 = 1.0 - 0.1 * 1.0 = 0.9

    p.grad = np.array([1.0])
    optimizer.step()
    # v2 = 0.9 * 1.0 + 1.0 = 1.9, p2 = 0.9 - 0.1 * 1.9 = 0.71

    np.testing.assert_allclose(p.data, np.array([0.71]))
    np.testing.assert_allclose(optimizer.v[0], np.array([1.9]))


def test_beta_zero_behaves_like_sgd() -> None:
    """При beta = 0 Momentum эквивалентен SGD."""
    rng = np.random.default_rng(0)
    init = rng.standard_normal(5)

    p_mom = Parameter(init.copy())
    p_sgd = Parameter(init.copy())
    optimizer_mom = Momentum([p_mom], lr=0.1, beta=0.0)
    optimizer_sgd = SGD([p_sgd], lr=0.1)

    for _ in range(5):
        grad = rng.standard_normal(5)
        p_mom.grad = grad.copy()
        p_sgd.grad = grad.copy()
        optimizer_mom.step()
        optimizer_sgd.step()

    np.testing.assert_allclose(p_mom.data, p_sgd.data)


# ---------- Пропуск None-градиентов ----------

def test_step_skips_none_grad() -> None:
    p = Parameter(np.array([1.0, 2.0]))
    optimizer = Momentum([p], lr=0.5)

    optimizer.step()

    np.testing.assert_array_equal(p.data, np.array([1.0, 2.0]))
    # буфер тоже остался нулевым
    np.testing.assert_array_equal(optimizer.v[0], np.array([0.0, 0.0]))


def test_step_mixed_none_and_real_grad() -> None:
    p_no_grad = Parameter(np.array([1.0]))
    p_with_grad = Parameter(np.array([1.0]))
    p_with_grad.grad = np.array([0.1])

    optimizer = Momentum([p_no_grad, p_with_grad], lr=0.5)
    optimizer.step()

    np.testing.assert_array_equal(p_no_grad.data, np.array([1.0]))
    np.testing.assert_array_equal(optimizer.v[0], np.array([0.0]))
    np.testing.assert_allclose(p_with_grad.data, np.array([0.95]))
    np.testing.assert_allclose(optimizer.v[1], np.array([0.1]))


# ---------- zero_grad (наследование) ----------

def test_zero_grad_resets_grads_but_not_buffers() -> None:
    """zero_grad обнуляет градиенты, но буферы v сохраняются."""
    p = Parameter(np.array([1.0]))
    p.grad = np.array([0.5])
    optimizer = Momentum([p], lr=0.1)

    optimizer.step()
    v_before = optimizer.v[0].copy()

    optimizer.zero_grad()

    assert p.grad is None
    np.testing.assert_array_equal(optimizer.v[0], v_before)


def test_zero_grad_only_touches_own_params() -> None:
    p_in = Parameter(np.array([1.0]))
    p_in.grad = np.array([0.1])
    p_out = Parameter(np.array([2.0]))
    p_out.grad = np.array([0.2])

    optimizer = Momentum([p_in], lr=0.1)
    optimizer.zero_grad()

    assert p_in.grad is None
    assert p_out.grad is not None


# ---------- In-place ----------

def test_param_data_updated_in_place() -> None:
    p = Parameter(np.array([1.0, 2.0]))
    p.grad = np.array([0.1, 0.2])
    data_before = p.data
    optimizer = Momentum([p], lr=0.1)

    optimizer.step()

    assert p.data is data_before


def test_velocity_updated_in_place() -> None:
    p = Parameter(np.array([1.0, 2.0]))
    p.grad = np.array([0.1, 0.2])
    optimizer = Momentum([p], lr=0.1)
    v_before = optimizer.v[0]

    optimizer.step()

    assert optimizer.v[0] is v_before


# ---------- Сходимость ----------

def test_converges_on_quadratic() -> None:
    """f(p) = 0.5 * ||p||^2, grad = p. Momentum должен дойти близко к нулю."""
    p = Parameter(np.array([1.0, -2.0, 3.0]))
    optimizer = Momentum([p], lr=0.05, beta=0.9)

    for _ in range(200):
        p.grad = p.data.copy()
        optimizer.step()

    assert np.allclose(p.data, 0.0, atol=1e-4)


def test_faster_than_sgd_on_ravine() -> None:
    """На овражной функции f(x, y) = 0.5 * (x^2 + 100 * y^2) Momentum
    должен сходиться быстрее SGD при одинаковых lr и числе шагов."""
    init = np.array([10.0, 10.0])
    grad_scale = np.array([1.0, 100.0])  # ∂f/∂x = x, ∂f/∂y = 100*y

    p_mom = Parameter(init.copy())
    p_sgd = Parameter(init.copy())
    optimizer_mom = Momentum([p_mom], lr=0.005, beta=0.9)
    optimizer_sgd = SGD([p_sgd], lr=0.005)

    for _ in range(100):
        p_mom.grad = p_mom.data * grad_scale
        p_sgd.grad = p_sgd.data * grad_scale
        optimizer_mom.step()
        optimizer_sgd.step()

    loss_mom = float((p_mom.data ** 2 * np.array([1.0, 100.0])).sum() / 2)
    loss_sgd = float((p_sgd.data ** 2 * np.array([1.0, 100.0])).sum() / 2)

    assert loss_mom < loss_sgd


# ---------- Сквозной тест с моделью ----------

def test_multiple_steps_decrease_loss() -> None:
    """Несколько шагов Momentum уменьшают MSE на игрушечной задаче."""
    from numpy_nn.nn import Linear, MSELoss, Sequential

    rng = np.random.default_rng(0)
    model = Sequential(Linear(4, 1))
    criterion = MSELoss()
    optimizer = Momentum(model.parameters(), lr=0.05, beta=0.9)

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
