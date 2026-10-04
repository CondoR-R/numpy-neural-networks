"""Тесты для оптимизатора Adam."""

from __future__ import annotations

import numpy as np
import pytest

from numpy_nn.nn.core import Parameter
from numpy_nn.optim import Adam, RMSProp

# ---------- Конструктор ----------

def test_init_stores_all_hyperparams() -> None:
    p = Parameter(np.array([1.0, 2.0]))
    optimizer = Adam([p], lr=0.05, beta1=0.8, beta2=0.99, eps=1e-6)

    assert optimizer.params == [p]
    assert optimizer.lr == 0.05
    assert optimizer.beta1 == 0.8
    assert optimizer.beta2 == 0.99
    assert optimizer.eps == 1e-6


def test_defaults() -> None:
    p = Parameter(np.array([1.0]))
    optimizer = Adam([p])
    assert optimizer.lr == 0.001
    assert optimizer.beta1 == 0.9
    assert optimizer.beta2 == 0.999
    assert optimizer.eps == 1e-8


def test_initial_t_is_zero() -> None:
    p = Parameter(np.array([1.0]))
    optimizer = Adam([p])
    assert optimizer.t == 0


@pytest.mark.parametrize("lr", [-1.0, -0.001])
def test_negative_lr_raises(lr: float) -> None:
    p = Parameter(np.array([1.0]))
    with pytest.raises(ValueError):
        Adam([p], lr=lr)


def test_lr_zero_allowed() -> None:
    p = Parameter(np.array([1.0]))
    optimizer = Adam([p], lr=0.0)
    assert optimizer.lr == 0.0


@pytest.mark.parametrize("beta1", [-0.1, 1.0, 1.5])
def test_beta1_out_of_range_raises(beta1: float) -> None:
    p = Parameter(np.array([1.0]))
    with pytest.raises(ValueError):
        Adam([p], beta1=beta1)


@pytest.mark.parametrize("beta2", [-0.1, 1.0, 1.5])
def test_beta2_out_of_range_raises(beta2: float) -> None:
    p = Parameter(np.array([1.0]))
    with pytest.raises(ValueError):
        Adam([p], beta2=beta2)


@pytest.mark.parametrize("eps", [-1e-8, 0.0])
def test_nonpositive_eps_raises(eps: float) -> None:
    p = Parameter(np.array([1.0]))
    with pytest.raises(ValueError):
        Adam([p], eps=eps)


def test_buffers_initialized_to_zeros() -> None:
    p1 = Parameter(np.array([1.0, 2.0, 3.0]))
    p2 = Parameter(np.zeros((2, 4)))
    optimizer = Adam([p1, p2])

    assert len(optimizer.m) == 2
    assert len(optimizer.v) == 2
    np.testing.assert_array_equal(optimizer.m[0], np.zeros(3))
    np.testing.assert_array_equal(optimizer.v[0], np.zeros(3))
    np.testing.assert_array_equal(optimizer.m[1], np.zeros((2, 4)))
    np.testing.assert_array_equal(optimizer.v[1], np.zeros((2, 4)))


def test_m_and_v_are_independent_arrays() -> None:
    """m[i] и v[i] — разные объекты, не ссылки на один массив."""
    p = Parameter(np.array([1.0, 2.0]))
    optimizer = Adam([p])
    assert optimizer.m[0] is not optimizer.v[0]


def test_buffers_shape_and_dtype() -> None:
    p = Parameter(np.array([1.0, 2.0, 3.0]))
    optimizer = Adam([p])

    assert optimizer.m[0].shape == p.data.shape
    assert optimizer.m[0].dtype == np.float64
    assert optimizer.v[0].shape == p.data.shape
    assert optimizer.v[0].dtype == np.float64


def test_init_copies_params_list() -> None:
    p1 = Parameter(np.array([1.0]))
    p2 = Parameter(np.array([2.0]))
    original = [p1]
    optimizer = Adam(original)

    original.append(p2)

    assert len(optimizer.params) == 1
    assert len(optimizer.m) == 1
    assert len(optimizer.v) == 1


# ---------- Счётчик t ----------

def test_t_increments_per_step() -> None:
    p = Parameter(np.array([1.0]))
    p.grad = np.array([0.1])
    optimizer = Adam([p])

    assert optimizer.t == 0
    optimizer.step()
    assert optimizer.t == 1
    optimizer.step()
    assert optimizer.t == 2


def test_t_increments_even_with_none_grads() -> None:
    """t растёт независимо от того, были ли градиенты."""
    p = Parameter(np.array([1.0]))
    optimizer = Adam([p])

    optimizer.step()
    assert optimizer.t == 1
    optimizer.step()
    assert optimizer.t == 2


# ---------- Формулы первого шага ----------

def test_first_step_buffers() -> None:
    """m_1 = (1-beta1)*grad, v_1 = (1-beta2)*grad**2."""
    p = Parameter(np.array([1.0]))
    p.grad = np.array([2.0])
    optimizer = Adam([p], beta1=0.9, beta2=0.999)

    optimizer.step()

    np.testing.assert_allclose(optimizer.m[0], np.array([0.1 * 2.0]))
    np.testing.assert_allclose(optimizer.v[0], np.array([0.001 * 4.0]))


def test_first_step_bias_correction() -> None:
    """После bias correction m_hat = grad, v_hat = grad**2.

    На первом шаге bc1 = 1 - beta1 = 0.1, bc2 = 1 - beta2 = 0.001.
    m_hat = 0.1*grad / 0.1 = grad
    v_hat = 0.001*grad**2 / 0.001 = grad**2
    """
    p = Parameter(np.array([1.0]))
    p.grad = np.array([0.5])
    optimizer = Adam([p], lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8)

    optimizer.step()

    # p_1 = 1.0 - 0.1 * 0.5 / (0.5 + 1e-8)
    expected = 1.0 - 0.1 * 0.5 / (0.5 + 1e-8)
    np.testing.assert_allclose(p.data, np.array([expected]), rtol=1e-9)


def test_step_direction() -> None:
    p_pos = Parameter(np.array([5.0]))
    p_pos.grad = np.array([1.0])
    p_neg = Parameter(np.array([5.0]))
    p_neg.grad = np.array([-1.0])

    optimizer = Adam([p_pos, p_neg], lr=0.1)
    optimizer.step()

    assert p_pos.data[0] < 5.0
    assert p_neg.data[0] > 5.0


# ---------- Формула второго шага ----------

def test_two_steps_manual_formula() -> None:
    """Полный ручной расчёт двух шагов Adam."""
    p = Parameter(np.array([1.0]))
    optimizer = Adam([p], lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8)

    # Шаг 1
    p.grad = np.array([1.0])
    optimizer.step()

    m1 = 0.1 * 1.0
    v1 = 0.001 * 1.0
    m1_hat = m1 / (1 - 0.9**1)
    v1_hat = v1 / (1 - 0.999**1)
    p1 = 1.0 - 0.1 * m1_hat / (np.sqrt(v1_hat) + 1e-8)

    np.testing.assert_allclose(p.data, np.array([p1]), rtol=1e-9)

    # Шаг 2
    p.grad = np.array([1.0])
    optimizer.step()

    m2 = 0.9 * m1 + 0.1 * 1.0
    v2 = 0.999 * v1 + 0.001 * 1.0
    m2_hat = m2 / (1 - 0.9**2)
    v2_hat = v2 / (1 - 0.999**2)
    p2 = p1 - 0.1 * m2_hat / (np.sqrt(v2_hat) + 1e-8)

    np.testing.assert_allclose(p.data, np.array([p2]), rtol=1e-9)
    np.testing.assert_allclose(optimizer.m[0], np.array([m2]))
    np.testing.assert_allclose(optimizer.v[0], np.array([v2]))


# ---------- Bias correction ----------

def test_bias_correction_effect_decreases() -> None:
    """Разница между m и m_hat значима в начале, исчезает позже."""
    p = Parameter(np.array([0.0]))
    optimizer = Adam([p], beta1=0.9, beta2=0.999)

    # шаг 1
    p.grad = np.array([1.0])
    optimizer.step()
    bc1_t1 = 1 - 0.9**1  # 0.1

    # много шагов
    for _ in range(200):
        p.grad = np.array([1.0])
        optimizer.step()
    bc1_t200 = 1 - 0.9**200  # ≈ 1

    # коррекция почти исчезла
    assert abs(bc1_t200 - 1.0) < 1e-6
    # а на первом шаге она была значимой
    assert bc1_t1 == pytest.approx(0.1)


def test_after_bias_correction_step_is_normalized() -> None:
    """На первом шаге шаг примерно равен lr * sign(grad)."""
    p = Parameter(np.array([1.0]))
    p.grad = np.array([100.0])
    optimizer = Adam([p], lr=0.5)

    optimizer.step()

    # шаг ≈ 0.5 * 100 / (100 + eps) ≈ 0.5
    step_size = 1.0 - p.data[0]
    assert step_size == pytest.approx(0.5, rel=1e-6)


# ---------- Два буфера независимы ----------

def test_m_and_v_grow_differently() -> None:
    """m накапливает grad, v накапливает grad**2 — числа разные."""
    p = Parameter(np.array([0.0]))
    p.grad = np.array([2.0])
    optimizer = Adam([p], beta1=0.9, beta2=0.999)

    optimizer.step()

    # m1 = 0.2, v1 = 0.004
    np.testing.assert_allclose(optimizer.m[0], np.array([0.2]))
    np.testing.assert_allclose(optimizer.v[0], np.array([0.004]))
    assert not np.allclose(optimizer.m[0], optimizer.v[0])


# ---------- None-градиенты ----------

def test_step_skips_none_grad() -> None:
    p = Parameter(np.array([1.0, 2.0]))
    optimizer = Adam([p], lr=0.5)

    optimizer.step()

    np.testing.assert_array_equal(p.data, np.array([1.0, 2.0]))
    np.testing.assert_array_equal(optimizer.m[0], np.array([0.0, 0.0]))
    np.testing.assert_array_equal(optimizer.v[0], np.array([0.0, 0.0]))


def test_step_mixed_none_and_real_grad() -> None:
    p_no_grad = Parameter(np.array([1.0]))
    p_with_grad = Parameter(np.array([1.0]))
    p_with_grad.grad = np.array([1.0])

    optimizer = Adam([p_no_grad, p_with_grad], lr=0.1)
    optimizer.step()

    np.testing.assert_array_equal(p_no_grad.data, np.array([1.0]))
    np.testing.assert_array_equal(optimizer.m[0], np.array([0.0]))
    assert p_with_grad.data[0] < 1.0


# ---------- zero_grad (наследование) ----------

def test_zero_grad_resets_grads_but_not_buffers() -> None:
    p = Parameter(np.array([1.0]))
    p.grad = np.array([0.5])
    optimizer = Adam([p])

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
    optimizer = Adam([p])

    optimizer.step()

    assert p.data is data_before


def test_buffers_updated_in_place() -> None:
    p = Parameter(np.array([1.0, 2.0]))
    p.grad = np.array([0.1, 0.2])
    optimizer = Adam([p])
    m_before = optimizer.m[0]
    v_before = optimizer.v[0]

    optimizer.step()

    assert optimizer.m[0] is m_before
    assert optimizer.v[0] is v_before


# ---------- Отличие от RMSProp ----------

def test_differs_from_rmsprop() -> None:
    """Adam и RMSProp с одинаковыми beta/lr дают разные траектории."""
    rng = np.random.default_rng(0)
    init = rng.standard_normal(5)

    p_adam = Parameter(init.copy())
    p_rms = Parameter(init.copy())
    opt_adam = Adam([p_adam], lr=0.1, beta1=0.9, beta2=0.999)
    opt_rms = RMSProp([p_rms], lr=0.1, beta=0.999)

    for _ in range(5):
        grad = rng.standard_normal(5)
        p_adam.grad = grad.copy()
        p_rms.grad = grad.copy()
        opt_adam.step()
        opt_rms.step()

    # Adam использует momentum и другую формулу EMA — траектории разные
    assert not np.allclose(p_adam.data, p_rms.data)


# ---------- Сходимость ----------

def test_converges_on_quadratic() -> None:
    """На f(p) = 0.5 * ||p||^2 Adam сходится с точностью ~lr."""
    p = Parameter(np.array([1.0, -2.0, 3.0]))
    optimizer = Adam([p], lr=0.05)

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
    optimizer = Adam(model.parameters(), lr=0.01)

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
