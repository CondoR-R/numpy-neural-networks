"""
Демонстрация нейрона МакКаллока-Питтса.
Показывает, как одним MP-нейроном реализовать логические функции
AND, OR, NOT, и как из трёх нейронов собрать XOR.
"""

from __future__ import annotations

import numpy as np

from numpy_nn.nn import MPNeuron


def print_truth_table_2(name: str, neuron: MPNeuron) -> None:
    """Печатает таблицу истинности для нейрона с двумя входами."""
    print(f"\n{name}")
    print(f"{'x1':>3} {'x2':>3} | out")
    print("-" * 14)
    for x1 in (0, 1):
        for x2 in (0, 1):
            x = np.array([x1, x2], dtype=np.float64)
            out = int(neuron(x))
            print(f"{x1:>3} {x2:>3} | {out:>3}")


def print_truth_table_1(name: str, neuron: MPNeuron) -> None:
    """Печатает таблицу истинности для нейрона с одним входом."""
    print(f"\n{name}")
    print(f"{'x':>3} | out")
    print("-" * 9)
    for x1 in (0, 1):
        x = np.array([x1], dtype=np.float64)
        out = int(neuron(x))
        print(f"{x1:>3} | {out:>3}")


def xor(x: np.ndarray) -> int:
    """XOR через композицию трёх MP-нейронов.

    XOR(x1, x2) = (x1 OR x2) AND NOT(x1 AND x2)
    """
    or_neuron = MPNeuron(weights=np.array([1.0, 1.0]), threshold=1.0)
    nand_neuron = MPNeuron(weights=np.array([-1.0, -1.0]), threshold=-1.0)
    and_neuron = MPNeuron(weights=np.array([1.0, 1.0]), threshold=2.0)

    h1 = or_neuron(x)
    h2 = nand_neuron(x)
    hidden = np.array([h1, h2], dtype=np.float64)
    return int(and_neuron(hidden))


def main() -> None:
    # Одиночные логические функции
    and_neuron = MPNeuron(weights=np.array([1.0, 1.0]), threshold=2.0)
    or_neuron = MPNeuron(weights=np.array([1.0, 1.0]), threshold=1.0)
    not_neuron = MPNeuron(weights=np.array([-1.0]), threshold=0.0)

    print("=" * 40)
    print("Одиночные MP-нейроны")
    print("=" * 40)

    print_truth_table_2("AND (w=[1,1], threshold=2)", and_neuron)
    print_truth_table_2("OR  (w=[1,1], threshold=1)", or_neuron)
    print_truth_table_1("NOT (w=[-1], threshold=0)", not_neuron)

    # XOR из трёх нейронов
    print("\n" + "=" * 40)
    print("XOR из трёх MP-нейронов")
    print("=" * 40)
    print(f"\n{'x1':>3} {'x2':>3} | xor")
    print("-" * 14)
    for x1 in (0, 1):
        for x2 in (0, 1):
            x = np.array([x1, x2], dtype=np.float64)
            print(f"{x1:>3} {x2:>3} | {xor(x):>3}")

    print("\nXOR = (x1 OR x2) AND NOT(x1 AND x2)")
    print("В сеть это превращается так:")
    print("  слой 1: OR и NAND (оба получают x1, x2)")
    print("  слой 2: AND поверх выходов OR и NAND")


if __name__ == "__main__":
    main()
