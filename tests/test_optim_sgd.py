"""Тесты для оптимизатора SGD."""

from __future__ import annotations

import numpy as np

from numpy_nn.nn.core import Parameter
from numpy_nn.optim import SGD

# ---------- Конструктор ----------

def test_init_stores_params_and_lr() -> None:
    p = Parameter(np.array([1.0, 2.0]))
    optimizer = SGD([p], lr=0.1)

    assert optimizer.params == [p]
    assert optimizer.lr == 0.1


def test_init_copies_params_list() -> None:
    """Изменение внешнего списка не влияет на оптимизатор."""
    p1 = Parameter(np.array([1.0]))
    p2 = Parameter(np.array([2.0]))
    original = [p1]
    optimizer = SGD(original, lr=0.1)

    original.append(p2)

    assert len(optimizer.params) == 1
    assert optimizer.params[0] is p1


# ---------- step: обновление ----------

def test_step_updates_single_param() -> None:
    p = Parameter(np.array([1.0, 2.0, 3.0]))
    p.grad = np.array([0.1, 0.2, 0.3])
    optimizer = SGD([p], lr=0.5)

    optimizer.step()

    expected = np.array([1.0 - 0.05, 2.0 - 0.10, 3.0 - 0.15])
    np.testing.assert_allclose(p.data, expected)


def test_step_updates_multiple_params() -> None:
    p1 = Parameter(np.array([1.0, 2.0]))
    p1.grad = np.array([0.5, 0.5])
    p2 = Parameter(np.array([10.0]))
    p2.grad = np.array([1.0])
    optimizer = SGD([p1, p2], lr=0.1)

    optimizer.step()

    np.testing.assert_allclose(p1.data, np.array([0.95, 1.95]))
    np.testing.assert_allclose(p2.data, np.array([9.9]))


def test_step_direction() -> None:
    """Градиент > 0 → параметр уменьшается. Градиент < 0 → растёт."""
    p_pos = Parameter(np.array([5.0]))
    p_pos.grad = np.array([1.0])
    p_neg = Parameter(np.array([5.0]))
    p_neg.grad = np.array([-1.0])

    optimizer = SGD([p_pos, p_neg], lr=0.1)
    optimizer.step()

    assert p_pos.data[0] < 5.0
    assert p_neg.data[0] > 5.0


def test_step_preserves_dtype() -> None:
    p = Parameter(np.array([1.0, 2.0]))
    p.grad = np.array([0.1, 0.2])
    optimizer = SGD([p], lr=0.1)

    optimizer.step()

    assert p.data.dtype == np.float64


def test_step_updates_in_place() -> None:
    """Ссылка на data не меняется, меняется содержимое."""
    p = Parameter(np.array([1.0, 2.0]))
    p.grad = np.array([0.1, 0.2])
    data_before = p.data
    optimizer = SGD([p], lr=0.1)

    optimizer.step()

    assert p.data is data_before


# ---------- step: None-градиенты ----------

def test_step_skips_none_grad() -> None:
    p = Parameter(np.array([1.0, 2.0]))
    # grad остаётся None
    optimizer = SGD([p], lr=0.5)

    optimizer.step()

    np.testing.assert_array_equal(p.data, np.array([1.0, 2.0]))


def test_step_mixed_none_and_real_grad() -> None:
    """Один параметр обновляется, другой — нет."""
    p_no_grad = Parameter(np.array([1.0]))
    p_with_grad = Parameter(np.array([1.0]))
    p_with_grad.grad = np.array([0.1])

    optimizer = SGD([p_no_grad, p_with_grad], lr=0.5)
    optimizer.step()

    np.testing.assert_array_equal(p_no_grad.data, np.array([1.0]))
    np.testing.assert_allclose(p_with_grad.data, np.array([0.95]))


def test_step_empty_params() -> None:
    """Пустой список параметров — не ошибка."""
    optimizer = SGD([], lr=0.1)
    optimizer.step()  # ничего не должно произойти


# ---------- lr = 0 ----------

def test_lr_zero_does_not_change_params() -> None:
    p = Parameter(np.array([1.0, 2.0]))
    p.grad = np.array([0.5, 0.5])
    optimizer = SGD([p], lr=0.0)

    optimizer.step()

    np.testing.assert_array_equal(p.data, np.array([1.0, 2.0]))


# ---------- zero_grad (наследование от Optimizer) ----------

def test_zero_grad_resets_grads() -> None:
    p1 = Parameter(np.array([1.0]))
    p1.grad = np.array([0.1])
    p2 = Parameter(np.array([2.0]))
    p2.grad = np.array([0.2])
    optimizer = SGD([p1, p2], lr=0.1)

    optimizer.zero_grad()

    assert p1.grad is None
    assert p2.grad is None


def test_zero_grad_only_touches_own_params() -> None:
    """Параметры, не переданные оптимизатору, не обнуляются."""
    p_in = Parameter(np.array([1.0]))
    p_in.grad = np.array([0.1])
    p_out = Parameter(np.array([2.0]))
    p_out.grad = np.array([0.2])

    optimizer = SGD([p_in], lr=0.1)
    optimizer.zero_grad()

    assert p_in.grad is None
    assert p_out.grad is not None


# ---------- Сходимость на квадратичной функции ----------

def test_converges_on_quadratic() -> None:
    """f(p) = 0.5 * ||p||^2, grad = p. После N шагов параметры близки к 0."""
    p = Parameter(np.array([1.0, -2.0, 3.0]))
    optimizer = SGD([p], lr=0.1)

    for _ in range(200):
        p.grad = p.data.copy()
        optimizer.step()

    assert np.allclose(p.data, 0.0, atol=1e-8)


def test_quadratic_step_after_one_iteration() -> None:
    """После одного шага p_1 = p_0 - lr * p_0 = (1 - lr) * p_0."""
    p = Parameter(np.array([1.0, 2.0, 3.0]))
    p.grad = p.data.copy()
    optimizer = SGD([p], lr=0.3)

    optimizer.step()

    expected = 0.7 * np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(p.data, expected)


# ---------- Сквозной тест с моделью ----------

def test_step_with_model_reduces_loss() -> None:
    """Один шаг SGD уменьшает MSE loss на игрушечной задаче."""
    from numpy_nn.nn import Linear, MSELoss, Sequential

    rng = np.random.default_rng(0)
    model = Sequential(Linear(3, 1))
    criterion = MSELoss()
    optimizer = SGD(model.parameters(), lr=0.05)

    x = rng.standard_normal((10, 3))
    y = rng.standard_normal((10, 1))

    # один шаг
    optimizer.zero_grad()
    pred = model(x)
    loss_before = criterion(pred, y)
    dpred = criterion.backward()
    model.backward(dpred)
    optimizer.step()

    # второй forward с обновлёнными весами
    loss_after = criterion(model(x), y)

    assert loss_after < loss_before


def test_multiple_steps_decrease_loss() -> None:
    """Несколько шагов SGD монотонно уменьшают MSE."""
    from numpy_nn.nn import Linear, MSELoss, Sequential

    rng = np.random.default_rng(1)
    model = Sequential(Linear(4, 1))
    criterion = MSELoss()
    optimizer = SGD(model.parameters(), lr=0.01)

    x = rng.standard_normal((20, 4))
    y = rng.standard_normal((20, 1))

    losses = []
    for _ in range(20):
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        losses.append(loss)
        dpred = criterion.backward()
        model.backward(dpred)
        optimizer.step()

    # loss на последней итерации меньше, чем в начале
    assert losses[-1] < losses[0]
