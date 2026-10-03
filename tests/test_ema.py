"""Тесты для экспоненциального скользящего среднего (EMA)."""

from __future__ import annotations

import numpy as np
import pytest

from numpy_nn.utils.ema import EMA

# ---------- Конструктор ----------

def test_default_beta() -> None:
    ema = EMA()
    assert ema.beta == 0.9


def test_beta_zero_allowed() -> None:
    ema = EMA(beta=0.0)
    assert ema.beta == 0.0


def test_beta_close_to_one_allowed() -> None:
    ema = EMA(beta=0.999)
    assert ema.beta == 0.999


@pytest.mark.parametrize("beta", [-0.1, -1.0, 1.0, 1.5, 2.0])
def test_beta_out_of_range_raises(beta: float) -> None:
    with pytest.raises(ValueError):
        EMA(beta=beta)


def test_initial_state_is_none() -> None:
    ema = EMA()
    assert ema.s is None
    assert ema.t == 0


# ---------- value до update ----------

def test_value_before_update_raises() -> None:
    ema = EMA()
    with pytest.raises(RuntimeError):
        _ = ema.value


# ---------- Скаляры ----------

def test_first_update_bias_corrected_equals_x() -> None:
    """
    После первого update value с bias correction = x.

    Без bias correction s_1 = (1 - beta) * x_1, что искажает первое
    значение. Bias correction возвращает x_1.
    """
    ema = EMA(beta=0.9)
    ema.update(5.0)
    assert ema.value == pytest.approx(5.0)


def test_second_update_matches_manual_formula() -> None:
    """Проверка формулы на двух шагах."""
    ema = EMA(beta=0.9)
    ema.update(1.0)
    ema.update(3.0)

    # s_1 = 0.1 * 1.0 = 0.1
    # s_2 = 0.9 * 0.1 + 0.1 * 3.0 = 0.09 + 0.3 = 0.39
    # s_hat_2 = 0.39 / (1 - 0.9^2) = 0.39 / 0.19 ≈ 2.0526...
    expected_s = 0.39
    expected_hat = expected_s / (1 - 0.9 ** 2)
    assert ema.value == pytest.approx(expected_hat)


def test_many_updates_approach_mean() -> None:
    """
    При большом числе шагов EMA сходится к среднему константной
    последовательности."""
    ema = EMA(beta=0.9)
    for _ in range(200):
        ema.update(7.0)
    assert ema.value == pytest.approx(7.0, rel=1e-6)


def test_beta_zero_returns_last_value() -> None:
    """При beta = 0 EMA равна последнему значению."""
    ema = EMA(beta=0.0)
    ema.update(1.0)
    ema.update(5.0)
    ema.update(-3.0)
    assert ema.value == pytest.approx(-3.0)


def test_accepts_int() -> None:
    """Целые числа принимаются и приводятся к float."""
    ema = EMA(beta=0.9)
    ema.update(3)
    assert ema.value == pytest.approx(3.0)


def test_counter_increments() -> None:
    ema = EMA()
    assert ema.t == 0
    ema.update(1.0)
    assert ema.t == 1
    ema.update(2.0)
    assert ema.t == 2
    ema.update(3.0)
    assert ema.t == 3


# ---------- Массивы ----------

def test_array_shape_preserved() -> None:
    ema = EMA(beta=0.9)
    x = np.array([1.0, 2.0, 3.0])
    ema.update(x)
    assert ema.value.shape == (3,)


def test_2d_array() -> None:
    ema = EMA(beta=0.9)
    x = np.random.randn(4, 5)
    ema.update(x)
    assert ema.value.shape == (4, 5)


def test_array_first_update_equals_x() -> None:
    """После первого update bias-corrected value = x поэлементно."""
    ema = EMA(beta=0.9)
    x = np.array([1.0, 2.0, 3.0])
    ema.update(x)
    np.testing.assert_allclose(ema.value, x)


def test_array_manual_formula() -> None:
    """Проверка формулы на массивах через ручной расчёт."""
    ema = EMA(beta=0.8)
    x1 = np.array([1.0, 2.0])
    x2 = np.array([3.0, 4.0])

    ema.update(x1)
    ema.update(x2)

    s1 = 0.2 * x1
    s2 = 0.8 * s1 + 0.2 * x2
    s2_hat = s2 / (1 - 0.8 ** 2)

    np.testing.assert_allclose(ema.value, s2_hat)


# ---------- Ошибки ----------

def test_shape_mismatch_raises() -> None:
    ema = EMA(beta=0.9)
    ema.update(np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError):
        ema.update(np.array([1.0, 2.0]))


def test_shape_mismatch_different_ndim_raises() -> None:
    ema = EMA(beta=0.9)
    ema.update(np.zeros((3, 4)))
    with pytest.raises(ValueError):
        ema.update(np.zeros((12,)))


def test_shape_mismatch_message_contains_shapes() -> None:
    """Сообщение об ошибке содержит обе формы."""
    ema = EMA(beta=0.9)
    ema.update(np.zeros((3,)))
    with pytest.raises(ValueError, match=r"\(3,\)"):
        ema.update(np.zeros((5,)))


# ---------- Внутреннее состояние ----------

def test_state_updated_in_place() -> None:
    """После update состояние — тот же массив, а не новый."""
    ema = EMA(beta=0.9)
    ema.update(np.array([1.0, 2.0]))
    s_before = ema.s
    ema.update(np.array([3.0, 4.0]))
    # если бы self.s пересоздавался, ссылка поменялась бы
    assert ema.s is s_before


def test_update_does_not_mutate_input() -> None:
    """update не должен мутировать переданный массив."""
    ema = EMA(beta=0.9)
    x1 = np.array([1.0, 2.0, 3.0])
    x2 = np.array([4.0, 5.0, 6.0])
    x2_copy = x2.copy()

    ema.update(x1)
    ema.update(x2)

    np.testing.assert_array_equal(x2, x2_copy)


# ---------- Свойства последовательности ----------

def test_ema_lags_behind_rising_sequence() -> None:
    """На возрастающей последовательности EMA отстаёт от последнего
    значения."""
    ema = EMA(beta=0.9)
    for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
        ema.update(v)
    # value должен быть больше 1.0, но меньше 5.0
    assert 1.0 < ema.value < 5.0


def test_ema_smooths_noise() -> None:
    """Сглаженное значение лежит в диапазоне входных."""
    rng = np.random.default_rng(0)
    values = rng.standard_normal(100) * 10
    ema = EMA(beta=0.9)
    for v in values:
        ema.update(float(v))
    assert values.min() <= ema.value <= values.max()
