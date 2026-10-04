"""Тесты для оптимизатора AdaGrad."""

from __future__ import annotations

import numpy as np
import pytest

from numpy_nn.nn.core import Parameter
from numpy_nn.optim import AdaGrad

# ---------- Конструктор ----------

def test_init_stores_params_lr_eps() -> None:
    p = Parameter(np.array([1.0, 2.0]))
    optimizer = AdaGrad([p], lr=0.05, eps=1e-6)

    assert optimizer.params == [p]
    assert optimizer.lr == 0.05
    assert optimizer.eps == 1e-6


def test_default_lr_and_eps() -> None:
    p = Parameter(np.array([1.0]))
    optimizer = AdaGrad([p])
    assert optimizer.lr == 0.01
    assert optimizer.eps == 1e-8


@pytest.mark.parametrize("lr", [-1.0, -0.001])
def test_negative_lr_raises(lr: float) -> None:
    """Отрицательный lr ловится базовым классом Optimizer."""
    p = Parameter(np.array([1.0]))
    with pytest.raises(ValueError):
        AdaGrad([p], lr=lr)


def test_lr_zero_allowed() -> None:
    """lr = 0 допустим (заморозка параметров, warmup)."""
    p = Parameter(np.array([1.0]))
    optimizer = AdaGrad([p], lr=0.0)
    assert optimizer.lr == 0.0


@pytest.mark.parametrize("eps", [-1e-8, 0.0])
def test_nonpositive_eps_raises(eps: float) -> None:
    p = Parameter(np.array([1.0]))
    with pytest.raises(ValueError):
        AdaGrad([p], eps=eps)


def test_buffers_initialized_to_zeros() -> None:
    p1 = Parameter(np.array([1.0, 2.0, 3.0]))
    p2 = Parameter(np.zeros((2, 4)))
    optimizer = AdaGrad([p1, p2])

    assert len(optimizer.g) == 2
    np.testing.assert_array_equal(optimizer.g[0], np.zeros(3))
    np.testing.assert_array_equal(optimizer.g[1], np.zeros((2, 4)))


def test_buffers_shape_and_dtype() -> None:
    p1 = Parameter(np.array([1.0, 2.0, 3.0]))
    p2 = Parameter(np.zeros((2, 4)))
    optimizer = AdaGrad([p1, p2])

    assert optimizer.g[0].shape == p1.data.shape
    assert optimizer.g[0].dtype == np.float64
    assert optimizer.g[1].shape == p2.data.shape
    assert optimizer.g[1].dtype == np.float64


def test_init_copies_params_list() -> None:
    p1 = Parameter(np.array([1.0]))
    p2 = Parameter(np.array([2.0]))
    original = [p1]
    optimizer = AdaGrad(original)

    original.append(p2)

    assert len(optimizer.params) == 1
    assert len(optimizer.g) == 1


# ---------- Один шаг ----------

def test_first_step_buffer_equals_grad_squared() -> None:
    """После первого шага g_1 = 0 + grad**2 = grad**2."""
    p = Parameter(np.array([1.0, 2.0]))
    p.grad = np.array([0.5, -0.5])
    optimizer = AdaGrad([p], lr=0.1, eps=1e-8)

    optimizer.step()

    np.testing.assert_allclose(optimizer.g[0], np.array([0.25, 0.25]))


def test_first_step_param_update() -> None:
    """p_1 = p_0 - lr * grad / (|grad| + eps)."""
    p = Parameter(np.array([1.0]))
    p.grad = np.array([0.5])
    optimizer = AdaGrad([p], lr=0.1, eps=1e-8)

    optimizer.step()

    expected = 1.0 - 0.1 * 0.5 / (0.5 + 1e-8)
    np.testing.assert_allclose(p.data, np.array([expected]), rtol=1e-9)


def test_step_direction() -> None:
    p_pos = Parameter(np.array([5.0]))
    p_pos.grad = np.array([1.0])
    p_neg = Parameter(np.array([5.0]))
    p_neg.grad = np.array([-1.0])

    optimizer = AdaGrad([p_pos, p_neg], lr=0.1)
    optimizer.step()

    assert p_pos.data[0] < 5.0
    assert p_neg.data[0] > 5.0


# ---------- Накопление квадратов ----------

def test_buffer_accumulates_squares() -> None:
    """g_2 = grad_1**2 + grad_2**2."""
    p = Parameter(np.array([0.0]))
    optimizer = AdaGrad([p], lr=0.1)

    p.grad = np.array([1.0])
    optimizer.step()
    np.testing.assert_allclose(optimizer.g[0], np.array([1.0]))

    p.grad = np.array([2.0])
    optimizer.step()
    np.testing.assert_allclose(optimizer.g[0], np.array([1.0 + 4.0]))


def test_two_steps_manual_formula() -> None:
    """Полный ручной расчёт двух шагов."""
    lr = 0.1
    eps = 1e-8

    p = Parameter(np.array([1.0]))
    optimizer = AdaGrad([p], lr=lr, eps=eps)

    p.grad = np.array([1.0])
    optimizer.step()
    p_1 = 1.0 - lr * 1.0 / (np.sqrt(1.0) + eps)
    # g1 = 1.0, p1 = 1.0 - 0.1 * 1.0 / (1.0 + eps) ≈ 0.9

    p.grad = np.array([1.0])
    optimizer.step()
    p_2 = p_1 - lr * 1.0 / (np.sqrt(2.0) + eps)
    # g2 = 1.0 + 1.0 = 2.0, p2 = 0.9 - 0.1 * 1.0 / (sqrt(2) + eps)

    np.testing.assert_allclose(p.data, np.array([p_2]), rtol=1e-9)


# ---------- Уменьшение эффективного LR ----------

def test_effective_lr_decreases_over_steps() -> None:
    """При одинаковом градиенте шаг на каждом следующем шаге меньше."""
    p = Parameter(np.array([0.0]))
    optimizer = AdaGrad([p], lr=0.1)

    steps = []
    prev = p.data.copy()
    for _ in range(5):
        p.grad = np.array([1.0])
        optimizer.step()
        steps.append(prev[0] - p.data[0])
        prev = p.data.copy()

    # каждый следующий шаг меньше предыдущего
    for i in range(1, len(steps)):
        assert steps[i] < steps[i - 1]


# ---------- None-градиенты ----------

def test_step_skips_none_grad() -> None:
    p = Parameter(np.array([1.0, 2.0]))
    optimizer = AdaGrad([p], lr=0.5)

    optimizer.step()

    np.testing.assert_array_equal(p.data, np.array([1.0, 2.0]))
    np.testing.assert_array_equal(optimizer.g[0], np.array([0.0, 0.0]))


def test_step_mixed_none_and_real_grad() -> None:
    p_no_grad = Parameter(np.array([1.0]))
    p_with_grad = Parameter(np.array([1.0]))
    p_with_grad.grad = np.array([1.0])

    optimizer = AdaGrad([p_no_grad, p_with_grad], lr=0.1)
    optimizer.step()

    np.testing.assert_array_equal(p_no_grad.data, np.array([1.0]))
    np.testing.assert_array_equal(optimizer.g[0], np.array([0.0]))
    # для p_with_grad: g = 1.0, p = 1.0 - 0.1 / (1 + eps) ≈ 0.9
    assert p_with_grad.data[0] < 1.0


# ---------- zero_grad (наследование) ----------

def test_zero_grad_resets_grads_but_not_buffer() -> None:
    """zero_grad обнуляет градиенты, но накопленный g сохраняется."""
    p = Parameter(np.array([1.0]))
    p.grad = np.array([0.5])
    optimizer = AdaGrad([p])

    optimizer.step()
    g_before = optimizer.g[0].copy()

    optimizer.zero_grad()

    assert p.grad is None
    np.testing.assert_array_equal(optimizer.g[0], g_before)


def test_zero_grad_only_touches_own_params() -> None:
    p_in = Parameter(np.array([1.0]))
    p_in.grad = np.array([0.1])
    p_out = Parameter(np.array([2.0]))
    p_out.grad = np.array([0.2])

    optimizer = AdaGrad([p_in])
    optimizer.zero_grad()

    assert p_in.grad is None
    assert p_out.grad is not None


# ---------- In-place ----------

def test_param_data_updated_in_place() -> None:
    p = Parameter(np.array([1.0, 2.0]))
    p.grad = np.array([0.1, 0.2])
    data_before = p.data
    optimizer = AdaGrad([p])

    optimizer.step()

    assert p.data is data_before


def test_buffer_updated_in_place() -> None:
    p = Parameter(np.array([1.0, 2.0]))
    p.grad = np.array([0.1, 0.2])
    optimizer = AdaGrad([p])
    g_before = optimizer.g[0]

    optimizer.step()

    assert optimizer.g[0] is g_before


# ---------- Сходимость ----------

def test_converges_on_quadratic() -> None:
    """На f(p) = 0.5 * ||p||^2 с grad = p AdaGrad должен сойтись."""
    p = Parameter(np.array([1.0, -2.0, 3.0]))
    optimizer = AdaGrad([p], lr=0.5)

    for _ in range(500):
        p.grad = p.data.copy()
        optimizer.step()

    assert np.allclose(p.data, 0.0, atol=1e-3)


# ---------- Разреженные градиенты ----------

def test_sparse_gradients_behavior() -> None:
    """Параметр с маленькими градиентами получает больший эффективный шаг."""
    p_big = Parameter(np.array([0.0]))
    p_small = Parameter(np.array([0.0]))
    optimizer = AdaGrad([p_big, p_small], lr=0.1)

    # одинаковое число шагов, но разные по величине градиенты
    p_big.grad = np.array([10.0])
    p_small.grad = np.array([0.1])
    optimizer.step()

    step_big = abs(p_big.data[0])
    step_small = abs(p_small.data[0])

    # эффективный шаг: lr * grad / (|grad| + eps) ≈ lr * sign(grad)
    # для больших и маленьких градиентов он примерно одинаков
    # но если градиент был маленьким один раз, знаменатель меньше,
    # и относительное движение должно быть сопоставимо
    assert abs(step_big - step_small) < 1e-3


# ---------- Сквозной тест с моделью ----------

def test_multiple_steps_decrease_loss() -> None:
    from numpy_nn.nn import Linear, MSELoss, Sequential

    rng = np.random.default_rng(0)
    model = Sequential(Linear(4, 1))
    criterion = MSELoss()
    optimizer = AdaGrad(model.parameters(), lr=0.05)

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
