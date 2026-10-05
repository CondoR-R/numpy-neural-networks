"""
Сравнение оптимизаторов на MNIST.

Обучает одну и ту же модель всеми оптимизаторами библиотеки
при нескольких learning rate, сохраняет результаты в JSON,
строит графики и пишет markdown-отчёт.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml

from numpy_nn.nn import CrossEntropyLoss, Linear, ReLU, Sequential
from numpy_nn.optim import (
    SGD,
    AdaGrad,
    Adam,
    AdamW,
    Momentum,
    Nadam,
    Nesterov,
    RMSProp,
)


# ---------- Гиперпараметры эксперимента ----------

SEED = 42
EPOCHS = 5
BATCH_SIZE = 128
VAL_SIZE = 5000

FIGURES_DIR = Path("figures")
REPORTS_DIR = Path("reports")
RESULTS_PATH = REPORTS_DIR / "compare_optimizers_results.json"
REPORT_PATH = REPORTS_DIR / "compare_optimizers.md"


# Формат строки: (имя, класс, kwargs, список learning rate)
OPTIMIZER_SPECS: list[tuple[str, Any, dict[str, float], list[float]]] = [
    ("SGD", SGD, {}, [1e-2, 3e-2, 1e-1]),
    ("Momentum", Momentum, {"beta": 0.9}, [1e-2, 3e-2, 1e-1]),
    ("Nesterov", Nesterov, {"beta": 0.9}, [1e-2, 3e-2, 1e-1]),
    ("AdaGrad", AdaGrad, {}, [1e-3, 1e-2, 1e-1]),
    ("RMSProp", RMSProp, {"beta": 0.9}, [1e-3, 1e-2, 1e-1]),
    ("Adam", Adam, {}, [1e-4, 1e-3, 1e-2]),
    ("AdamW", AdamW, {"weight_decay": 0.01}, [1e-4, 1e-3, 1e-2]),
    ("Nadam", Nadam, {}, [1e-4, 1e-3, 1e-2]),
]


# Цвета для каждого оптимизатора — чтобы совпадали во всех графиках
COLORS: dict[str, str] = {
    "SGD": "tab:blue",
    "Momentum": "tab:orange",
    "Nesterov": "tab:green",
    "AdaGrad": "tab:red",
    "RMSProp": "tab:purple",
    "Adam": "tab:brown",
    "AdamW": "tab:pink",
    "Nadam": "tab:gray",
}


# ---------- Данные ----------

def load_and_prepare_data() -> tuple[
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
]:
    """Загружает MNIST, нормализует, разбивает на train/val/test.

    Returns:
        Кортеж (X_train, y_train, X_val, y_val, X_test, y_test).
        X — float64 формы (N, 784) в диапазоне [0, 1].
        y — int64 формы (N,).
    """
    print("Loading MNIST (первый запуск может занять минуту)...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X = np.asarray(mnist.data, dtype=np.float64) / 255.0
    y = np.asarray(mnist.target, dtype=np.int64)

    # перемешиваем и режем на train/val/test
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]

    X_test = X[-10_000:]
    y_test = y[-10_000:]

    X_val = X[:VAL_SIZE]
    y_val = y[:VAL_SIZE]

    X_train = X[VAL_SIZE:-10_000]
    y_train = y[VAL_SIZE:-10_000]


    print(f"  train: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test


# ---------- Модель ----------

def build_model() -> Sequential:
    """Свежая модель с одинаковой инициализацией для всех прогонов."""
    np.random.seed(SEED)
    return Sequential(
        Linear(784, 128),
        ReLU(),
        Linear(128, 64),
        ReLU(),
        Linear(64, 10),
    )


# ---------- Один прогон ----------

def train_one(
    name: str,
    opt_cls: Any,
    opt_kwargs: dict[str, float],
    lr: float,
    data: tuple[np.ndarray, ...],
) -> dict[str, Any]:
    """Обучает модель заданным оптимизатором и возвращает метрики."""
    X_train, y_train, X_val, y_val, X_test, y_test = data

    model = build_model()
    criterion = CrossEntropyLoss()
    optimizer = opt_cls(model.parameters(), lr=lr, **opt_kwargs)

    n_train = len(X_train)
    train_losses: list[float] = []
    val_losses: list[float] = []
    val_accuracies: list[float] = []

    start = time.perf_counter()

    for epoch in range(EPOCHS):
        perm = np.random.permutation(n_train)
        batch_losses: list[float] = []

        for i in range(0, n_train, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb = X_train[idx]
            yb = y_train[idx]

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            dlogits = criterion.backward()
            model.backward(dlogits)
            optimizer.step()

            batch_losses.append(loss)

        train_loss = float(np.mean(batch_losses))

        # метрики на val
        val_logits = model(X_val)
        val_loss = criterion(val_logits, y_val)
        val_acc = float((val_logits.argmax(axis=1) == y_val).mean())

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(
            f"[{name:>8} lr={lr:.0e}] epoch {epoch + 1}/{EPOCHS}: "
            f"train_loss={train_loss:.4f} val_acc={val_acc:.4f}"
        )

    elapsed = time.perf_counter() - start

    # тест
    test_logits = model(X_test)
    test_acc = float((test_logits.argmax(axis=1) == y_test).mean())

    # эпох до достижения val_acc >= 0.95
    epochs_to_95 = -1
    for i, acc in enumerate(val_accuracies):
        if acc >= 0.95:
            epochs_to_95 = i + 1
            break

    return {
        "optimizer": name,
        "lr": lr,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
        "test_accuracy": test_acc,
        "time_seconds": elapsed,
        "epochs_to_95": epochs_to_95,
    }


# ---------- Все прогоны ----------

def run_all_experiments(data: tuple[np.ndarray, ...]) -> list[dict[str, Any]]:
    """Прогоняет все пары (оптимизатор, lr)."""
    results: list[dict[str, Any]] = []
    total = sum(len(spec[3]) for spec in OPTIMIZER_SPECS)
    counter = 0

    for name, opt_cls, kwargs, lrs in OPTIMIZER_SPECS:
        for lr in lrs:
            counter += 1
            print(f"\n[{counter}/{total}] {name} lr={lr:.0e}")
            result = train_one(name, opt_cls, kwargs, lr, data)
            results.append(result)

    return results


# ---------- Сохранение результатов ----------

def _to_python(obj: Any) -> Any:
    """Рекурсивно приводит numpy-типы к Python — для JSON."""
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_python(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def save_results(results: list[dict[str, Any]]) -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(_to_python(results), indent=2),
        encoding="utf-8",
    )
    print(f"Saved results to {RESULTS_PATH}")


# ---------- Графики ----------

def _best_run_per_optimizer(
    results: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Для каждого оптимизатора — его лучший прогон по финальной val_acc."""
    best: dict[str, dict[str, Any]] = {}
    for r in results:
        name = r["optimizer"]
        final_acc = r["val_accuracies"][-1]
        if name not in best or final_acc > best[name]["val_accuracies"][-1]:
            best[name] = r
    return best


def plot_train_loss(results: list[dict[str, Any]]) -> None:
    best = _best_run_per_optimizer(results)
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, EPOCHS + 1)
    for name, r in best.items():
        ax.plot(epochs, r["train_losses"], label=name,
                color=COLORS[name], marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train loss")
    ax.set_title("Train loss vs epoch (best lr per optimizer)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(
        FIGURES_DIR / "compare_optimizers_train_loss.png",
        dpi=100, bbox_inches="tight",
    )
    plt.close(fig)


def plot_val_accuracy(results: list[dict[str, Any]]) -> None:
    best = _best_run_per_optimizer(results)
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, EPOCHS + 1)
    for name, r in best.items():
        ax.plot(epochs, r["val_accuracies"], label=name,
                color=COLORS[name], marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val accuracy")
    ax.set_title("Val accuracy vs epoch (best lr per optimizer)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(
        FIGURES_DIR / "compare_optimizers_val_accuracy.png",
        dpi=100, bbox_inches="tight",
    )
    plt.close(fig)


def plot_lr_sensitivity(results: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, _, _, _ in OPTIMIZER_SPECS:
        rs = sorted(
            (r for r in results if r["optimizer"] == name),
            key=lambda r: r["lr"],
        )
        lrs = [r["lr"] for r in rs]
        accs = [r["test_accuracy"] for r in rs]
        ax.plot(lrs, accs, label=name, color=COLORS[name], marker="o")
    ax.set_xscale("log")
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Test accuracy vs learning rate")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    fig.savefig(
        FIGURES_DIR / "compare_optimizers_lr_sensitivity.png",
        dpi=100, bbox_inches="tight",
    )
    plt.close(fig)


def plot_time_vs_accuracy(results: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, _, _, _ in OPTIMIZER_SPECS:
        rs = [r for r in results if r["optimizer"] == name]
        times = [r["time_seconds"] for r in rs]
        accs = [r["test_accuracy"] for r in rs]
        ax.scatter(times, accs, label=name, color=COLORS[name], s=60)
    ax.set_xlabel("Training time (s)")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Time vs test accuracy")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(
        FIGURES_DIR / "compare_optimizers_time_vs_accuracy.png",
        dpi=100, bbox_inches="tight",
    )
    plt.close(fig)


def plot_all(results: list[dict[str, Any]]) -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    plot_train_loss(results)
    plot_val_accuracy(results)
    plot_lr_sensitivity(results)
    plot_time_vs_accuracy(results)
    print(f"Saved plots to {FIGURES_DIR}")


# ---------- Отчёт ----------

def write_report(results: list[dict[str, Any]]) -> None:
    lines: list[str] = []
    lines.append("# Сравнение оптимизаторов на MNIST")
    lines.append("")
    lines.append(f"- Эпох: {EPOCHS}")
    lines.append(f"- Batch size: {BATCH_SIZE}")
    lines.append("- Архитектура: 784 → 128 → 64 → 10 (ReLU)")
    lines.append("- AdamW запускался с weight_decay=0.01")
    lines.append("")
    lines.append("## Результаты")
    lines.append("")
    lines.append("| Optimizer | lr | Эпох до 95% | Val acc | Test acc | Время, с |")
    lines.append("|---|---|---|---|---|---|")

    for r in sorted(results, key=lambda x: (x["optimizer"], x["lr"])):
        lines.append(
            f"| {r['optimizer']} | {r['lr']:.0e} | {r['epochs_to_95']} "
            f"| {r['val_accuracies'][-1]:.4f} | {r['test_accuracy']:.4f} "
            f"| {r['time_seconds']:.1f} |"
        )

    best = max(results, key=lambda r: r["test_accuracy"])
    fastest = min(results, key=lambda r: r["time_seconds"])

    lines.append("")
    lines.append("## Выводы")
    lines.append("")
    lines.append(
        f"- Лучший по test accuracy: **{best['optimizer']}** "
        f"(lr={best['lr']:.0e}) — {best['test_accuracy']:.4f}"
    )
    lines.append(
        f"- Самый быстрый: **{fastest['optimizer']}** "
        f"(lr={fastest['lr']:.0e}) — {fastest['time_seconds']:.1f} с"
    )

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved report to {REPORT_PATH}")


# ---------- Точка входа ----------

def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    data = load_and_prepare_data()
    results = run_all_experiments(data)
    save_results(results)
    plot_all(results)
    write_report(results)

    print("\nDone.")


if __name__ == "__main__":
    main()
