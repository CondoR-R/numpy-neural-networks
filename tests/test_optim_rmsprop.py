"""Тесты для оптимизатора RMSProp."""

from __future__ import annotations

import numpy as np
import pytest

from numpy_nn.nn.core import Parameter
from numpy_nn.optim import AdaGrad, RMSProp

# ---------- Конструктор ----------

def test_init_stores_params_lr_beta_eps() -> None:
    p = Parameter(np.array([1.0, 2.0]))
    optimizer = RMSProp([p], lr=0.05, beta=0.8, eps=1e-6)

    assert optimizer.params == [p]
    assert optimizer.lr == 0.05
    assert optimizer.beta == 0.8
    assert optimizer.eps == 1e-6


def test_defaults() -> None:
    p = Parameter(np.array([1.0]))
    optimizer = RMSProp([p])
    assert optimizer.lr == 0.01
    assert optimizer.beta == 0.9
    assert optimizer.eps == 1e-8


@pytest.mark.parametrize("lr", [-1.0, -0.001])
def test_negative_lr_raises(lr: float) -> None:
    """Валидация lr наследуется от базового Optimizer."""
    p = Parameter(np.array([1.0]))
    with pytest.raises(ValueError):
        RMSProp([p], lr=lr)


def test_lr_zero_allowed() -> None:
    p = Parameter(np.array([1.0]))
    optimizer = RMSProp([p], lr=0.0)
    assert optimizer.lr == 0.0


@pytest.mark.parametrize("beta", [-0.1, 1.0, 1.5])
def test_beta_out_of_range_raises(beta: float) -> None:
    p = Parameter(np.array([1.0]))
    with pytest.raises(ValueError):
        RMSProp([p], beta=beta)


def test_beta_zero_allowed() -> None:
    p = Parameter(np.array([1.0]))
    optimizer = RMSProp([p], beta=0.0)
    assert optimizer.beta == 0.0


@pytest.mark.parametrize("eps", [-1e-8, 0.0])
def test_nonpositive_eps_raises(eps: float) -> None:
    p = Parameter(np.array([1.0]))
    with pytest.raises(ValueError):
        RMSProp([p], eps=eps)


def test_buffers_initialized_to_zeros() -> None:
    p1 = Parameter(np.array([1.0, 2.0, 3.0]))
    p2 = Parameter(np.zeros((2, 4)))
    optimizer = RMSProp([p1, p2])

    assert len(optimizer.v) == 2
    np.testing.assert_array_equal(optimizer.v[0], np.zeros(3))
    np.testing.assert_array_equal(optimizer.v[1], np.zeros((2, 4)))


def test_buffers_shape_and_dtype() -> None:
    p1 = Parameter(np.array([1.0, 2.0, 3.0]))
    p2 = Parameter(np.zeros((2, 4)))
    optimizer = RMSProp([p1, p2])

    assert optimizer.v[0].shape == p1.data.shape
    assert optimizer.v[0].dtype == np.float64
    assert optimizer.v[1].shape == p2.data.shape
    assert optimizer.v[1].dtype == np.float64


def test_init_copies_params_list() -> None:
    p1 = Parameter(np.array([1.0]))
    p2 = Parameter(np.array([2.0]))
    original = [p1]
    optimizer = RMSProp(original)

    original.append(p2)

    assert len(optimizer.params) == 1
    assert len(optimizer.v) == 1


# ---------- Один шаг ----------

def test_first_step_buffer_equals_grad_squared() -> None:
    """На первом шаге v_1 = 0 * beta + grad**2 = grad**2."""
    p = Parameter(np.array([1.0, 2.0]))
    p.grad = np.array([0.5, -0.5])
    optimizer = RMSProp([p], lr=0.1, beta=0.9, eps=1e-8)

    optimizer.step()

    np.testing.assert_allclose(optimizer.v[0], np.array([0.25, 0.25]))


def test_first_step_param_update() -> None:
    """p_1 = p_0 - lr * grad / (|grad| + eps)."""
    p = Parameter(np.array([1.0]))
    p.grad = np.array([0.5])
    optimizer = RMSProp([p], lr=0.1, beta=0.9, eps=1e-8)

    optimizer.step()

    expected = 1.0 - 0.1 * 0.5 / (0.5 + 1e-8)
    np.testing.assert_allclose(p.data, np.array([expected]), rtol=1e-9)


def test_step_direction() -> None:
    p_pos = Parameter(np.array([5.0]))
    p_pos.grad = np.array([1.0])
    p_neg = Parameter(np.array([5.0]))
    p_neg.grad = np.array([-1.0])

    optimizer = RMSProp([p_pos, p_neg], lr=0.1)
    optimizer.step()

    assert p_pos.data[0] < 5.0
    assert p_neg.data[0] > 5.0


# ---------- Накопление через EMA ----------

def test_buffer_uses_ema_not_sum() -> None:
    """v_2 = beta * v_1 + grad_2**2, а не v_1 + grad_2**2."""
    p = Parameter(np.array([0.0]))
    optimizer = RMSProp([p], lr=0.1, beta=0.9)

    p.grad = np.array([1.0])
    optimizer.step()
    np.testing.assert_allclose(optimizer.v[0], np.array([1.0]))

    p.grad = np.array([2.0])
    optimizer.step()
    # v_2 = 0.9 * 1.0 + 4.0 = 4.9
    np.testing.assert_allclose(optimizer.v[0], np.array([4.9]))


def test_two_steps_manual_formula() -> None:
    """Полный ручной расчёт двух шагов."""
    p = Parameter(np.array([1.0]))
    optimizer = RMSProp([p], lr=0.1, beta=0.9, eps=1e-8)

    p.grad = np.array([1.0])
    optimizer.step()
    # v1 = 1.0
    # p1 = 1.0 - 0.1 * 1.0 / (1.0 + 1e-8)

    p1_expected = 1.0 - 0.1 * 1.0 / (1.0 + 1e-8)

    p.grad = np.array([1.0])
    optimizer.step()
    # v2 = 0.9 * 1.0 + 1.0 = 1.9
    # p2 = p1 - 0.1 * 1.0 / (sqrt(1.9) + 1e-8)
    p2_expected = p1_expected - 0.1 * 1.0 / (np.sqrt(1.9) + 1e-8)

    np.testing.assert_allclose(p.data, np.array([p2_expected]), rtol=1e-9)
    np.testing.assert_allclose(optimizer.v[0], np.array([1.9]))


# ---------- Отличие от AdaGrad ----------

def test_differs_from_adagrad_after_second_step() -> None:
    """После первого шага v и g совпадают, после второго — расходятся."""
    p_rms = Parameter(np.array([1.0]))
    p_ada = Parameter(np.array([1.0]))
    opt_rms = RMSProp([p_rms], lr=0.1, beta=0.9, eps=1e-8)
    opt_ada = AdaGrad([p_ada], lr=0.1, eps=1e-8)

    # первый шаг — одинаково
    p_rms.grad = np.array([1.0])
    p_ada.grad = np.array([1.0])
    opt_rms.step()
    opt_ada.step()
    np.testing.assert_allclose(opt_rms.v[0], opt_ada.g[0])

    # второй шаг с другим градиентом — формулы расходятся
    p_rms.grad = np.array([2.0])
    p_ada.grad = np.array([2.0])
    opt_rms.step()
    opt_ada.step()
    assert not np.allclose(opt_rms.v[0], opt_ada.g[0])


# ---------- Стабилизация эффективного LR ----------

def test_effective_lr_stabilizes_not_decays_to_zero() -> None:
    """При одинаковом градиенте шаг стремится к пределу, а не к нулю."""
    p = Parameter(np.array([0.0]))
    optimizer = RMSProp([p], lr=0.1, beta=0.9)

    steps = []
    prev = p.data.copy()
    for _ in range(50):
        p.grad = np.array([1.0])
        optimizer.step()
        steps.append(prev[0] - p.data[0])
        prev = p.data.copy()

    # шаг монотонно стремится к пределу lr / sqrt(v_inf), где v_inf = 1/(1-beta) = 10
    # значит предельный шаг ≈ 0.1 / sqrt(10) ≈ 0.0316
    # последние шаги уже почти не меняются
    last_steps = steps[-5:]
    for s in last_steps:
        assert abs(s - last_steps[0]) < 1e-3


def test_beta_zero_no_smoothing() -> None:
    """При beta = 0 буфер равен grad**2 на каждом шаге."""
    p = Parameter(np.array([1.0]))
    optimizer = RMSProp([p], lr=0.1, beta=0.0, eps=1e-8)

    p.grad = np.array([2.0])
    optimizer.step()
    np.testing.assert_allclose(optimizer.v[0], np.array([4.0]))

    p.grad = np.array([3.0])
    optimizer.step()
    # при beta = 0 v не помнит предыдущее: v = 0 + 9 = 9
    np.testing.assert_allclose(optimizer.v[0], np.array([9.0]))


# ---------- None-градиенты ----------

def test_step_skips_none_grad() -> None:
    p = Parameter(np.array([1.0, 2.0]))
    optimizer = RMSProp([p], lr=0.5)

    optimizer.step()

    np.testing.assert_array_equal(p.data, np.array([1.0, 2.0]))
    np.testing.assert_array_equal(optimizer.v[0], np.array([0.0, 0.0]))


def test_step_mixed_none_and_real_grad() -> None:
    p_no_grad = Parameter(np.array([1.0]))
    p_with_grad = Parameter(np.array([1.0]))
    p_with_grad.grad = np.array([1.0])

    optimizer = RMSProp([p_no_grad, p_with_grad], lr=0.1)
    optimizer.step()

    np.testing.assert_array_equal(p_no_grad.data, np.array([1.0]))
    np.testing.assert_array_equal(optimizer.v[0], np.array([0.0]))
    assert p_with_grad.data[0] < 1.0


# ---------- zero_grad (наследование) ----------

def test_zero_grad_resets_grads_but_not_buffer() -> None:
    p = Parameter(np.array([1.0]))
    p.grad = np.array([0.5])
    optimizer = RMSProp([p])

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
    optimizer = RMSProp([p])

    optimizer.step()

    assert p.data is data_before


def test_buffer_updated_in_place() -> None:
    p = Parameter(np.array([1.0, 2.0]))
    p.grad = np.array([0.1, 0.2])
    optimizer = RMSProp([p])
    v_before = optimizer.v[0]

    optimizer.step()

    assert optimizer.v[0] is v_before


# ---------- Сходимость ----------

def test_converges_on_quadratic() -> None:
    p = Parameter(np.array([1.0, -2.0, 3.0]))
    optimizer = RMSProp([p], lr=0.1)

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
    optimizer = RMSProp(model.parameters(), lr=0.01)

    x = rng.standard_normal((20, 4))
    y = rng.standard_normal((20, 1))

    losses = []
    for _ in range(50):
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        losses.append(loss)
        dpred = criterion.backward()
        model.backward(dpred)
        optimizer.step()

    assert losses[-1] < losses[0]
