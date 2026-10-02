"""Тесты для нейрона МакКаллока-Питтса."""

from __future__ import annotations

import numpy as np
import pytest

from numpy_nn.nn import MPNeuron

# ---------- Вспомогательные фикстуры ----------

@pytest.fixture
def and_neuron() -> MPNeuron:
    return MPNeuron(weights=np.array([1.0, 1.0]), threshold=2.0)


@pytest.fixture
def or_neuron() -> MPNeuron:
    return MPNeuron(weights=np.array([1.0, 1.0]), threshold=1.0)


@pytest.fixture
def not_neuron() -> MPNeuron:
    return MPNeuron(weights=np.array([-1.0]), threshold=0.0)


# ---------- Логические функции ----------

@pytest.mark.parametrize(
    ("x", "expected"),
    [
        ([0.0, 0.0], 0),
        ([0.0, 1.0], 0),
        ([1.0, 0.0], 0),
        ([1.0, 1.0], 1),
    ],
)
def test_and(and_neuron: MPNeuron, x: list[float], expected: int) -> None:
    result = and_neuron(np.array(x))
    assert int(result) == expected


@pytest.mark.parametrize(
    ("x", "expected"),
    [
        ([0.0, 0.0], 0),
        ([0.0, 1.0], 1),
        ([1.0, 0.0], 1),
        ([1.0, 1.0], 1),
    ],
)
def test_or(or_neuron: MPNeuron, x: list[float], expected: int) -> None:
    result = or_neuron(np.array(x))
    assert int(result) == expected


@pytest.mark.parametrize(
    ("x", "expected"),
    [
        ([0.0], 1),
        ([1.0], 0),
    ],
)
def test_not(not_neuron: MPNeuron, x: list[float], expected: int) -> None:
    result = not_neuron(np.array(x))
    assert int(result) == expected


# ---------- Батчевый вход ----------

def test_batch_input(or_neuron: MPNeuron) -> None:
    x = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    result = or_neuron(x)
    np.testing.assert_array_equal(result, np.array([0, 1, 1, 1], dtype=np.uint8))


# ---------- Граница порога ----------

def test_threshold_boundary_hits() -> None:
    """Сумма ровно равна порогу → нейрон срабатывает (>=)."""
    neuron = MPNeuron(weights=np.array([1.0, 1.0]), threshold=2.0)
    assert int(neuron(np.array([1.0, 1.0]))) == 1


def test_threshold_just_below() -> None:
    """Сумма чуть меньше порога → нейрон не срабатывает."""
    neuron = MPNeuron(weights=np.array([1.0, 1.0]), threshold=2.5)
    assert int(neuron(np.array([1.0, 1.0]))) == 0


# ---------- Ошибки размерности ----------

def test_shape_mismatch_single(and_neuron: MPNeuron) -> None:
    with pytest.raises(ValueError):
        and_neuron(np.array([1.0, 0.0, 1.0]))


def test_shape_mismatch_batch(and_neuron: MPNeuron) -> None:
    x = np.zeros((3, 5))
    with pytest.raises(ValueError):
        and_neuron(x)


# ---------- dtype выхода ----------

def test_output_dtype(or_neuron: MPNeuron) -> None:
    result = or_neuron(np.array([1.0, 0.0]))
    assert result.dtype == np.uint8
