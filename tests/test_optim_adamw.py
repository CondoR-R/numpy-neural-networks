"""Тесты для оптимизатора AdamW."""

from __future__ import annotations

import numpy as np
import pytest

from numpy_nn.nn.core import Parameter
from numpy_nn.optim import Adam, AdamW

# ---------- Конструктор и наследование ----------

def test_adamw_is_adam_subclass() -> None:
    """AdamW наследуется от Adam — разделяет его поля состояния."""
    p = Parameter(np.array([1.0]))
    optimizer = AdamW([p])
    assert isinstance(optimizer, Adam)


def test_init_stores_weight_decay() -> None:
    p = Parameter(np.array([1.0]))
    optimizer = AdamW([p], weight_decay=0.05)
    assert optimizer.weight_decay == 0.05


def test_defaults() -> None:
    p = Parameter(np.array([1.0]))
    optimizer = AdamW([p])
    assert optimizer.lr == 0.001
    assert optimizer.beta1 == 0.9
    assert optimizer.beta2 == 0.999
    assert optimizer.eps == 1e-8
    assert optimizer.weight_decay == 0.01


def test_weight_decay_zero_allowed() -> None:
    p = Parameter(np.array([1.0]))
    optimizer = AdamW([p], weight_decay=0.0)
    assert optimizer.weight_decay == 0.0


@pytest.mark.parametrize("wd", [-0.1, -1.0])
def test_negative_weight_decay_raises(wd: float) -> None:
    p = Parameter(np.array([1.0]))
    with pytest.raises(ValueError):
        AdamW([p], weight_decay=wd)


def test_inherits_adam_validation() -> None:
    """Валидация beta1/beta2/eps/lr идёт через Adam.__init__."""
    p = Parameter(np.array([1.0]))
    with pytest.raises(ValueError):
        AdamW([p], beta1=1.5)
    with pytest.raises(ValueError):
        AdamW([p], beta2=-0.1)
    with pytest.raises(ValueError):
        AdamW([p], eps=0.0)
    with pytest.raises(ValueError):
        AdamW([p], lr=-0.001)


def test_buffers_initialized_to_zeros() -> None:
    p1 = Parameter(np.array([1.0, 2.0, 3.0]))
    p2 = Parameter(np.zeros((2, 4)))
    optimizer = AdamW([p1, p2])

    assert len(optimizer.m) == 2
    assert len(optimizer.v) == 2
    np.testing.assert_array_equal(optimizer.m[0], np.zeros(3))
    np.testing.assert_array_equal(optimizer.v[0], np.zeros(3))


def test_initial_t_is_zero() -> None:
    p = Parameter(np.array([1.0]))
    optimizer = AdamW([p])
    assert optimizer.t == 0


# ---------- weight_decay = 0 эквивалентен Adam ----------

def test_weight_decay_zero_equals_adam() -> None:
    """При weight_decay = 0 AdamW совпадает с Adam."""
    rng = np.random.default_rng(0)
    init = rng.standard_normal(5)

    p_adam = Parameter(init.copy())
    p_adamw = Parameter(init.copy())
    opt_adam = Adam([p_adam], lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8)
    opt_adamw = AdamW(
        [p_adamw], lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0
    )

    for _ in range(10):
        grad = rng.standard_normal(5)
        p_adam.grad = grad.copy()
        p_adamw.grad = grad.copy()
        opt_adam.step()
        opt_adamw.step()

    np.testing.assert_allclose(p_adamw.data, p_adam.data)
    np.testing.assert_allclose(opt_adamw.m[0], opt_adam.m[0])
    np.testing.assert_allclose(opt_adamw.v[0], opt_adam.v[0])


# ---------- weight_decay > 0 отличается от Adam ----------

def test_weight_decay_positive_differs_from_adam() -> None:
    """При weight_decay > 0 траектории Adam и AdamW расходятся."""
    rng = np.random.default_rng(1)
    init = rng.standard_normal(5)

    p_adam = Parameter(init.copy())
    p_adamw = Parameter(init.copy())
    opt_adam = Adam([p_adam], lr=0.1)
    opt_adamw = AdamW([p_adamw], lr=0.1, weight_decay=0.1)

    for _ in range(5):
        grad = rng.standard_normal(5)
        p_adam.grad = grad.copy()
        p_adamw.grad = grad.copy()
        opt_adam.step()
        opt_adamw.step()

    assert not np.allclose(p_adamw.data, p_adam.data)


# ---------- Формула первого шага ----------

def test_first_step_manual_formula() -> None:
    """Полный ручной расчёт первого шага AdamW."""
    p = Parameter(np.array([1.0]))
    p.grad = np.array([1.0])
    optimizer = AdamW(
        [p], lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.1
    )

    optimizer.step()

    # t = 1, bc1 = 0.1, bc2 = 0.001
    # m = 0.1, v = 0.001
    # m_hat = 1.0, v_hat = 1.0
    # decay: p *= (1 - 0.1 * 0.1) = 0.99
    # adam: p -= 0.1 * 1.0 / (1.0 + 1e-8)
    expected = 1.0 * 0.99 - 0.1 * 1.0 / (1.0 + 1e-8)
    np.testing.assert_allclose(p.data, np.array([expected]), rtol=1e-9)


def test_two_steps_manual_formula() -> None:
    """Полный ручной расчёт двух шагов AdamW."""
    p = Parameter(np.array([1.0]))
    optimizer = AdamW(
        [p], lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.1
    )

    p.grad = np.array([1.0])
    optimizer.step()

    # восстановим p1 из тех же формул
    m1 = 0.1 * 1.0
    v1 = 0.001 * 1.0
    m1_hat = m1 / 0.1
    v1_hat = v1 / 0.001
    p1 = 1.0 * 0.99 - 0.1 * m1_hat / (np.sqrt(v1_hat) + 1e-8)

    p.grad = np.array([1.0])
    optimizer.step()

    m2 = 0.9 * m1 + 0.1 * 1.0
    v2 = 0.999 * v1 + 0.001 * 1.0
    m2_hat = m2 / (1 - 0.9**2)
    v2_hat = v2 / (1 - 0.999**2)
    p2 = p1 * 0.99 - 0.1 * m2_hat / (np.sqrt(v2_hat) + 1e-8)

    np.testing.assert_allclose(p.data, np.array([p2]), rtol=1e-9)


# ---------- Decay работает даже при нулевом градиенте ----------

def test_decay_pulls_param_even_with_zero_grad() -> None:
    """При grad = 0 (не None) AdamW уменьшает |p|, а Adam — нет."""
    p_adam = Parameter(np.array([10.0]))
    p_adamw = Parameter(np.array([10.0]))
    p_adam.grad = np.array([0.0])
    p_adamw.grad = np.array([0.0])

    Adam([p_adam], lr=0.1).step()
    AdamW([p_adamw], lr=0.1, weight_decay=0.5).step()

    # Adam: m = 0, v = 0, обновление = 0, p остался 10
    np.testing.assert_allclose(p_adam.data, np.array([10.0]))

    # AdamW: decay сделал p *= (1 - 0.1 * 0.5) = 0.95, потом обновление 0
    np.testing.assert_allclose(p_adamw.data, np.array([9.5]))


def test_decay_reduces_absolute_value_after_gradient_steps() -> None:
    """После шагов с ненулевым градиентом AdamW сжимает |p| сильнее Adam."""
    p_adam = Parameter(np.array([1.0]))
    p_adamw = Parameter(np.array([1.0]))
    opt_adam = Adam([p_adam], lr=0.01)
    opt_adamw = AdamW([p_adamw], lr=0.01, weight_decay=0.5)

    for _ in range(20):
        p_adam.grad = np.array([0.0])
        p_adamw.grad = np.array([0.0])
        opt_adam.step()
        opt_adamw.step()

    # Adam не двигает параметр
    np.testing.assert_allclose(p_adam.data, np.array([1.0]))
    # AdamW сжимает экспоненциально
    assert p_adamw.data[0] < 1.0


# ---------- None-градиенты ----------

def test_none_grad_skipped_no_decay() -> None:
    """Параметр с grad is None не трогается вообще: ни decay, ни обновление."""
    p = Parameter(np.array([1.0, 2.0]))
    optimizer = AdamW([p], lr=0.5, weight_decay=0.5)

    optimizer.step()

    np.testing.assert_array_equal(p.data, np.array([1.0, 2.0]))
    np.testing.assert_array_equal(optimizer.m[0], np.array([0.0, 0.0]))
    np.testing.assert_array_equal(optimizer.v[0], np.array([0.0, 0.0]))


def test_mixed_none_and_zero_grad() -> None:
    """None-градиент: пропуск. Нулевой градиент: только decay."""
    p_none = Parameter(np.array([10.0]))
    p_zero = Parameter(np.array([10.0]))
    p_zero.grad = np.array([0.0])

    optimizer = AdamW([p_none, p_zero], lr=0.1, weight_decay=0.5)
    optimizer.step()

    # p_none не тронут
    np.testing.assert_allclose(p_none.data, np.array([10.0]))
    # p_zero сжат decay'ем
    np.testing.assert_allclose(p_zero.data, np.array([10.0 * 0.95]))


# ---------- Счётчик t ----------

def test_t_increments_per_step() -> None:
    p = Parameter(np.array([1.0]))
    p.grad = np.array([0.1])
    optimizer = AdamW([p])

    assert optimizer.t == 0
    optimizer.step()
    assert optimizer.t == 1
    optimizer.step()
    assert optimizer.t == 2


def test_t_increments_even_with_none_grads() -> None:
    p = Parameter(np.array([1.0]))
    optimizer = AdamW([p])

    optimizer.step()
    assert optimizer.t == 1
    optimizer.step()
    assert optimizer.t == 2


# ---------- zero_grad (наследование) ----------

def test_zero_grad_resets_grads_but_not_buffers() -> None:
    p = Parameter(np.array([1.0]))
    p.grad = np.array([0.5])
    optimizer = AdamW([p])

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
    optimizer = AdamW([p])

    optimizer.step()

    assert p.data is data_before


def test_buffers_updated_in_place() -> None:
    p = Parameter(np.array([1.0, 2.0]))
    p.grad = np.array([0.1, 0.2])
    optimizer = AdamW([p])
    m_before = optimizer.m[0]
    v_before = optimizer.v[0]

    optimizer.step()

    assert optimizer.m[0] is m_before
    assert optimizer.v[0] is v_before


# ---------- Сходимость ----------

def test_converges_on_quadratic() -> None:
    p = Parameter(np.array([1.0, -2.0, 3.0]))
    optimizer = AdamW([p], lr=0.05)

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
    optimizer = AdamW(model.parameters(), lr=0.01, weight_decay=0.01)

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
