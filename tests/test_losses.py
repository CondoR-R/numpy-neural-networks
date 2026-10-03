"""Тесты для функций потерь CrossEntropyLoss и MSELoss."""

from __future__ import annotations

import numpy as np
import pytest

from numpy_nn.nn.losses import CrossEntropyLoss, MSELoss

# ============================================================
# CrossEntropyLoss
# ============================================================

# ---------- Значения loss ----------

def test_ce_random_guess_is_log_c() -> None:
    """При нулевых логитах loss = log(C)."""
    criterion = CrossEntropyLoss()
    n, c = 5, 3
    logits = np.zeros((n, c))
    target = np.array([0, 1, 2, 0, 1])

    loss = criterion(logits, target)
    np.testing.assert_allclose(loss, np.log(c), rtol=1e-6)


def test_ce_confident_correct_is_near_zero() -> None:
    """При больших логитах для правильного класса loss близок к 0."""
    criterion = CrossEntropyLoss()
    logits = np.array([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0]])
    target = np.array([0, 1])

    loss = criterion(logits, target)
    assert loss < 1e-6


def test_ce_confident_wrong_is_large() -> None:
    """При больших логитах для неправильного класса loss большой."""
    criterion = CrossEntropyLoss()
    logits = np.array([[100.0, 0.0, 0.0]])
    target = np.array([1])

    loss = criterion(logits, target)
    assert loss > 50


def test_ce_reference_value() -> None:
    """Проверка на маленьком примере через ручной расчёт."""
    criterion = CrossEntropyLoss()
    logits = np.array([[1.0, 2.0, 3.0]])
    target = np.array([2])

    # softmax([1, 2, 3]) — считаем вручную
    e = np.exp(np.array([1.0, 2.0, 3.0]))
    p = e / e.sum()
    expected = -np.log(p[2])

    np.testing.assert_allclose(criterion(logits, target), expected, rtol=1e-12)


def test_ce_numerical_stability() -> None:
    """Большие логиты не должны давать nan или inf."""
    criterion = CrossEntropyLoss()
    logits = np.array([[1000.0, -1000.0, 0.0]])
    target = np.array([0])

    loss = criterion(logits, target)
    assert np.isfinite(loss)
    assert loss < 1e-6


# ---------- Backward ----------

def test_ce_backward_shape() -> None:
    criterion = CrossEntropyLoss()
    logits = np.random.randn(4, 5)
    target = np.array([0, 1, 2, 3])

    criterion(logits, target)
    dlogits = criterion.backward()
    assert dlogits.shape == (4, 5)


def test_ce_backward_requires_forward() -> None:
    criterion = CrossEntropyLoss()
    with pytest.raises(AssertionError):
        criterion.backward()


def test_ce_backward_rows_sum_to_zero() -> None:
    """Сумма градиента по классам для каждого примера равна 0.

    Свойство softmax: градиент CE по логитам всегда в сумме даёт ноль
    по оси классов. Это хорошая инварианта для проверки.
    """
    criterion = CrossEntropyLoss()
    logits = np.random.randn(4, 5)
    target = np.array([0, 1, 2, 3])

    criterion(logits, target)
    dlogits = criterion.backward()

    np.testing.assert_allclose(dlogits.sum(axis=1), np.zeros(4), atol=1e-12)


def test_ce_backward_at_correct_class_is_negative() -> None:
    """В позиции правильного класса градиент отрицательный.

    Формула (probs - one_hot) / N: в правильной позиции probs[y] - 1 < 0,
    потому что probs[y] < 1. Значит, градиент там отрицательный.
    """
    criterion = CrossEntropyLoss()
    logits = np.random.randn(3, 4)
    target = np.array([1, 2, 0])

    criterion(logits, target)
    dlogits = criterion.backward()

    for i, t in enumerate(target):
        assert dlogits[i, t] < 0


# ============================================================
# MSELoss
# ============================================================

# ---------- Значения loss ----------

def test_mse_zero_on_perfect_prediction() -> None:
    criterion = MSELoss()
    x = np.array([1.0, 2.0, 3.0])
    assert criterion(x, x) == 0.0


def test_mse_reference_value() -> None:
    """Проверка MSE через ручной расчёт."""
    criterion = MSELoss()
    pred = np.array([1.0, 2.0, 3.0])
    target = np.array([1.5, 2.0, 2.5])

    expected = np.mean((pred - target) ** 2)
    np.testing.assert_allclose(criterion(pred, target), expected, rtol=1e-12)


def test_mse_2d_shape() -> None:
    """MSE работает с массивами любой формы — усредняет по всем элементам."""
    criterion = MSELoss()
    pred = np.random.randn(3, 4)
    target = np.random.randn(3, 4)

    loss = criterion(pred, target)
    assert isinstance(loss, float)
    assert loss >= 0


# ---------- Backward ----------

def test_mse_backward_shape() -> None:
    criterion = MSELoss()
    pred = np.random.randn(3, 4)
    target = np.random.randn(3, 4)

    criterion(pred, target)
    dpred = criterion.backward()
    assert dpred.shape == (3, 4)


def test_mse_backward_requires_forward() -> None:
    criterion = MSELoss()
    with pytest.raises(AssertionError):
        criterion.backward()


def test_mse_backward_formula() -> None:
    """Backward совпадает с ручной формулой 2 * diff / n."""
    criterion = MSELoss()
    pred = np.array([1.0, 2.0, 3.0])
    target = np.array([1.5, 2.0, 2.5])

    criterion(pred, target)
    dpred = criterion.backward()

    expected = 2 * (pred - target) / pred.size
    np.testing.assert_allclose(dpred, expected, rtol=1e-12)


def test_mse_backward_zero_on_perfect_prediction() -> None:
    """Если pred == target, градиент = 0."""
    criterion = MSELoss()
    x = np.array([1.0, 2.0, 3.0])

    criterion(x, x)
    dpred = criterion.backward()
    np.testing.assert_array_equal(dpred, np.zeros(3))


# ============================================================
# Gradient checking
# ============================================================

def _num_grad_ce(
    logits: np.ndarray,
    target: np.ndarray,
    index: tuple[int, ...],
    eps: float = 1e-6,
) -> float:
    """Численный градиент CE по одному элементу logits."""
    criterion = CrossEntropyLoss()

    logits_plus = logits.copy()
    logits_plus[index] += eps
    l_plus = criterion(logits_plus, target)

    logits_minus = logits.copy()
    logits_minus[index] -= eps
    l_minus = criterion(logits_minus, target)

    return (l_plus - l_minus) / (2 * eps)


def test_ce_gradient_check() -> None:
    """Численный градиент CE совпадает с аналитическим."""
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((3, 4))
    target = np.array([0, 2, 3])

    criterion = CrossEntropyLoss()
    criterion(logits, target)
    dlogits = criterion.backward()

    for idx in np.ndindex(logits.shape):
        num = _num_grad_ce(logits, target, idx)
        analytic = dlogits[idx]
        assert abs(num - analytic) < 1e-6, (
            f"CE mismatch at {idx}: analytic={analytic}, numeric={num}"
        )


def _num_grad_mse(
    pred: np.ndarray,
    target: np.ndarray,
    index: tuple[int, ...],
    eps: float = 1e-6,
) -> float:
    """Численный градиент MSE по одному элементу pred."""
    criterion = MSELoss()

    pred_plus = pred.copy()
    pred_plus[index] += eps
    l_plus = criterion(pred_plus, target)

    pred_minus = pred.copy()
    pred_minus[index] -= eps
    l_minus = criterion(pred_minus, target)

    return (l_plus - l_minus) / (2 * eps)


def test_mse_gradient_check() -> None:
    """Численный градиент MSE совпадает с аналитическим."""
    rng = np.random.default_rng(1)
    pred = rng.standard_normal((3, 4))
    target = rng.standard_normal((3, 4))

    criterion = MSELoss()
    criterion(pred, target)
    dpred = criterion.backward()

    for idx in np.ndindex(pred.shape):
        num = _num_grad_mse(pred, target, idx)
        analytic = dpred[idx]
        assert abs(num - analytic) < 1e-6, (
            f"MSE mismatch at {idx}: analytic={analytic}, numeric={num}"
        )


# ============================================================
# Совместимость с моделью (сквозной сценарий)
# ============================================================

def test_ce_with_sequential_model() -> None:
    """Типичный сценарий: Sequential → logits → CE → backward → model.backward."""
    from numpy_nn.nn.activations import ReLU
    from numpy_nn.nn.layers import Linear
    from numpy_nn.nn.sequential import Sequential

    rng = np.random.default_rng(42)
    model = Sequential(
        Linear(4, 8),
        ReLU(),
        Linear(8, 3),
    )
    criterion = CrossEntropyLoss()

    x = rng.standard_normal((5, 4))
    target = np.array([0, 1, 2, 0, 1])

    logits = model(x)
    loss = criterion(logits, target)
    dlogits = criterion.backward()
    model.backward(dlogits)

    # после полного прохода у всех параметров есть градиент
    for p in model.parameters():
        assert p.grad is not None
        assert p.grad.shape == p.data.shape

    assert np.isfinite(loss)
    assert loss > 0
