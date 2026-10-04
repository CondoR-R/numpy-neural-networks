"""Тесты для оптимизатора Nadam."""

from __future__ import annotations

import numpy as np
import pytest

from numpy_nn.nn.core import Parameter
from numpy_nn.optim import Adam, Nadam

# ---------- Конструктор и наследование ----------


def test_nadam_is_adam_subclass() -> None:
    p = Parameter(np.array([1.0]))
    optimizer = Nadam([p])
    assert isinstance(optimizer, Adam)


def test_defaults() -> None:
    p = Parameter(np.array([1.0]))
    optimizer = Nadam([p])
    assert optimizer.lr == 0.001
    assert optimizer.beta1 == 0.9
    assert optimizer.beta2 == 0.999
    assert optimizer.eps == 1e-8


def test_initial_t_is_zero() -> None:
    p = Parameter(np.array([1.0]))
    optimizer = Nadam([p])
    assert optimizer.t == 0


def test_inherits_validation() -> None:
    """Валидация гиперпараметров идёт через Adam.__init__."""
    p = Parameter(np.array([1.0]))
    with pytest.raises(ValueError):
        Nadam([p], beta1=1.5)
    with pytest.raises(ValueError):
        Nadam([p], beta2=-0.1)
    with pytest.raises(ValueError):
        Nadam([p], eps=0.0)
    with pytest.raises(ValueError):
        Nadam([p], lr=-0.001)


def test_buffers_initialized_to_zeros() -> None:
    p1 = Parameter(np.array([1.0, 2.0, 3.0]))
    p2 = Parameter(np.zeros((2, 4)))
    optimizer = Nadam([p1, p2])

    assert len(optimizer.m) == 2
    assert len(optimizer.v) == 2
    np.testing.assert_array_equal(optimizer.m[0], np.zeros(3))
    np.testing.assert_array_equal(optimizer.v[0], np.zeros(3))


# ---------- Отличие от Adam ----------


def test_differs_from_adam() -> None:
    """Nadam с Nesterov-слагаемым даёт другие траектории, чем Adam."""
    rng = np.random.default_rng(0)
    init = rng.standard_normal(5)

    p_adam = Parameter(init.copy())
    p_nadam = Parameter(init.copy())
    opt_adam = Adam([p_adam], lr=0.1)
    opt_nadam = Nadam([p_nadam], lr=0.1)

    for _ in range(5):
        grad = rng.standard_normal(5)
        p_adam.grad = grad.copy()
        p_nadam.grad = grad.copy()
        opt_adam.step()
        opt_nadam.step()

    assert not np.allclose(p_adam.data, p_nadam.data)


# ---------- Формула первого шага ----------


def test_first_step_manual_formula() -> None:
    """Полный ручной расчёт первого шага Nadam."""
    p = Parameter(np.array([1.0]))
    p.grad = np.array([1.0])
    optimizer = Nadam([p], lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8)

    optimizer.step()

    # t = 1
    m1 = 0.1 * 1.0
    v1 = 0.001 * 1.0
    bc1 = 1 - 0.9**1
    bc2 = 1 - 0.999**1
    bc1_next = 1 - 0.9**2

    m_hat = m1 / bc1
    m_nesterov = 0.9 * m_hat + (1 - 0.9) * 1.0 / bc1_next

    expected = 1.0 - 0.1 * m_nesterov / (np.sqrt(v1 / bc2) + 1e-8)
    np.testing.assert_allclose(p.data, np.array([expected]), rtol=1e-9)


def test_two_steps_manual_formula() -> None:
    """Полный ручной расчёт двух шагов."""
    p = Parameter(np.array([1.0]))
    optimizer = Nadam([p], lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8)

    # Шаг 1
    p.grad = np.array([1.0])
    optimizer.step()

    m1 = 0.1 * 1.0
    v1 = 0.001 * 1.0
    bc1_1 = 1 - 0.9**1
    bc2_1 = 1 - 0.999**1
    bc1_next_1 = 1 - 0.9**2
    m_hat_1 = m1 / bc1_1
    mn_1 = 0.9 * m_hat_1 + 0.1 * 1.0 / bc1_next_1
    p1 = 1.0 - 0.1 * mn_1 / (np.sqrt(v1 / bc2_1) + 1e-8)

    np.testing.assert_allclose(p.data, np.array([p1]), rtol=1e-9)

    # Шаг 2
    p.grad = np.array([1.0])
    optimizer.step()

    m2 = 0.9 * m1 + 0.1 * 1.0
    v2 = 0.999 * v1 + 0.001 * 1.0
    bc1_2 = 1 - 0.9**2
    bc2_2 = 1 - 0.999**2
    bc1_next_2 = 1 - 0.9**3
    m_hat_2 = m2 / bc1_2
    mn_2 = 0.9 * m_hat_2 + 0.1 * 1.0 / bc1_next_2
    p2 = p1 - 0.1 * mn_2 / (np.sqrt(v2 / bc2_2) + 1e-8)

    np.testing.assert_allclose(p.data, np.array([p2]), rtol=1e-9)
    np.testing.assert_allclose(optimizer.m[0], np.array([m2]))
    np.testing.assert_allclose(optimizer.v[0], np.array([v2]))


def test_first_step_differs_from_adam_by_grad_term() -> None:
    """
    Nadam на первом шаге отличается от Adam дополнительным слагаемым.

    У Adam: p1 = 1 - lr * m_hat / (sqrt(v_hat) + eps).
    У Nadam: p1 = 1 - lr * (beta1 * m_hat + (1-beta1)*grad/bc1_next) / (...) .

    Nesterov-слагаемое добавляет положительный вклад, если grad > 0,
    поэтому шаг Nadam больше по модулю (параметр сдвигается дальше).
    """
    init = np.array([1.0])

    p_adam = Parameter(init.copy())
    p_adam.grad = np.array([1.0])
    p_nadam = Parameter(init.copy())
    p_nadam.grad = np.array([1.0])

    Adam([p_adam], lr=0.1).step()
    Nadam([p_nadam], lr=0.1).step()

    # для положительного градиента параметр уходит вниз, Nadam уходит дальше
    assert p_nadam.data[0] < p_adam.data[0]


# ---------- None-градиенты ----------


def test_step_skips_none_grad() -> None:
    p = Parameter(np.array([1.0, 2.0]))
    optimizer = Nadam([p], lr=0.5)

    optimizer.step()

    np.testing.assert_array_equal(p.data, np.array([1.0, 2.0]))
    np.testing.assert_array_equal(optimizer.m[0], np.array([0.0, 0.0]))
    np.testing.assert_array_equal(optimizer.v[0], np.array([0.0, 0.0]))


def test_step_mixed_none_and_real_grad() -> None:
    p_no_grad = Parameter(np.array([1.0]))
    p_with_grad = Parameter(np.array([1.0]))
    p_with_grad.grad = np.array([1.0])

    optimizer = Nadam([p_no_grad, p_with_grad], lr=0.1)
    optimizer.step()

    np.testing.assert_array_equal(p_no_grad.data, np.array([1.0]))
    np.testing.assert_array_equal(optimizer.m[0], np.array([0.0]))
    assert p_with_grad.data[0] < 1.0


# ---------- Счётчик t ----------


def test_t_increments_per_step() -> None:
    p = Parameter(np.array([1.0]))
    p.grad = np.array([0.1])
    optimizer = Nadam([p])

    assert optimizer.t == 0
    optimizer.step()
    assert optimizer.t == 1
    optimizer.step()
    assert optimizer.t == 2


def test_t_increments_even_with_none_grads() -> None:
    p = Parameter(np.array([1.0]))
    optimizer = Nadam([p])

    optimizer.step()
    assert optimizer.t == 1
    optimizer.step()
    assert optimizer.t == 2


# ---------- zero_grad (наследование) ----------


def test_zero_grad_resets_grads_but_not_buffers() -> None:
    p = Parameter(np.array([1.0]))
    p.grad = np.array([0.5])
    optimizer = Nadam([p])

    optimizer.step()
    m_before = optimizer.m[0].copy()
    v_before = optimizer.v[0].copy()

    optimizer.zero_grad()

    assert p.grad is None
    np.testing.assert_array_equal(optimizer.m[0], m_before)
    np.testing.assert_array_equal(optimizer.v[0], v_before)


# ---------- In-place ----------


def test_param_data_updated_in_place() -> None:
    p = Parameter(np.array([1.0, 2.0]))
    p.grad = np.array([0.1, 0.2])
    data_before = p.data
    optimizer = Nadam([p])

    optimizer.step()

    assert p.data is data_before


def test_buffers_updated_in_place() -> None:
    p = Parameter(np.array([1.0, 2.0]))
    p.grad = np.array([0.1, 0.2])
    optimizer = Nadam([p])
    m_before = optimizer.m[0]
    v_before = optimizer.v[0]

    optimizer.step()

    assert optimizer.m[0] is m_before
    assert optimizer.v[0] is v_before


# ---------- Сходимость ----------


def test_converges_on_quadratic() -> None:
    p = Parameter(np.array([1.0, -2.0, 3.0]))
    optimizer = Nadam([p], lr=0.05)

    for _ in range(500):
        p.grad = p.data.copy()
        optimizer.step()

    assert np.allclose(p.data, 0.0, atol=0.1)


# ---------- Сквозной тест с моделью ----------


def test_multiple_steps_decrease_loss() -> None:
    from numpy_nn.nn import Linear, MSELoss, Sequential

    rng = np.random.default_rng(0)
    model = Sequential(Linear(4, 1))
    criterion = MSELoss()
    optimizer = Nadam(model.parameters(), lr=0.01)

    x = rng.standard_normal((20, 4))
    y = rng.standard_normal((20, 1))

    losses = []
    for _ in range(100):
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        losses.append(loss)
        dpred = criterion.backward()
        model.backward(dpred)
        optimizer.step()

    assert losses[-1] < losses[0]
