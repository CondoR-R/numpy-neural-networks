# numpy-neural-networks

Учебная библиотека для глубокого обучения, реализованная с нуля на NumPy.
Цель — понять каждую формулу и каждый алгоритм через собственную реализацию,
без использования PyTorch, TensorFlow и других фреймворков.

## Установка

```bash
uv sync
```

## Проверка

```bash
uv run pytest
uv run ruff check .
uv run mypy src
```

## Что реализовано

### Ядро

- [x] `Parameter` — обёртка над обучаемым массивом (данные + градиент)
- [x] `Module` — базовый интерфейс для всех модулей
- [x] Прямой проход (`forward`)
- [x] Обратный проход (`backward`) без autograd
- [x] Gradient checking в тестах для всех слоёв и активаций

### Слои и модули

- [x] `Linear` — полносвязный слой с Xavier-инициализацией
- [x] `ReLU`, `Sigmoid`, `Tanh` — активации как модули
- [x] `Sequential` — контейнер для композиции слоёв
- [x] `CrossEntropyLoss` — кросс-энтропия с softmax внутри
- [x] `MSELoss` — среднеквадратичная ошибка

### Оптимизаторы

- [ ] SGD
- [ ] Momentum
- [ ] Nesterov Accelerated Momentum (NAG)
- [ ] AdaGrad
- [ ] RMSProp
- [ ] Adam
- [ ] AdamW
- [ ] Nadam

### Утилиты

- [ ] Экспоненциальное скользящее среднее (EMA)
- [ ] Learning rate schedulers
- [ ] Загрузчик MNIST / Fashion-MNIST

## Прогресс по темам

1. [x] Модель нейрона МакКаллока–Питтса (удалён из библиотеки, см. [`reports/`](reports/01_mp_neuron.md))
2. [x] Строение многослойного перцептрона
3. [x] Функции активации
4. [x] Прямой и обратный проход
5. [ ] Обучение нейронной сети (частично: forward/backward есть, тренировочный цикл — нет)
6. [ ] Оптимизаторы в глубоком обучении
7. [ ] Экспоненциальное скользящее среднее
8. [ ] SGD
9. [ ] Momentum
10. [ ] NAG
11. [ ] AdaGrad
12. [ ] RMSProp
13. [ ] Adam
14. [ ] AdamW
15. [ ] Nadam
16. [ ] Сравнение оптимизаторов

## Структура проекта

```
src/numpy_nn/
├── nn/         # слои, активации, контейнеры, функции потерь
│   ├── core.py        # Parameter, Module
│   ├── layers.py      # Linear
│   ├── activations.py # функции и модули активаций
│   ├── sequential.py  # Sequential
│   └── losses.py      # CrossEntropyLoss, MSELoss
├── optim/      # оптимизаторы (пока пусто)
├── utils/      # утилиты: EMA, schedulers, метрики (частично)
└── data/       # загрузчики датасетов (пока пусто)

tests/          # pytest-тесты
examples/       # примеры использования
notebooks/      # эксперименты
reports/        # конспекты по темам
figures/        # графики (не версионируются)
data/           # локальные датасеты (не версионируются)
```

## Использование

Пример: MLP для многоклассовой классификации.

```python
import numpy as np
from numpy_nn.nn import Linear, ReLU, Sequential, CrossEntropyLoss

model = Sequential(
    Linear(784, 128),
    ReLU(),
    Linear(128, 64),
    ReLU(),
    Linear(64, 10),
)

criterion = CrossEntropyLoss()

# forward
logits = model(x)
loss = criterion(logits, y)

# backward (градиенты пишутся в параметры)
dlogits = criterion.backward()
model.backward(dlogits)

# доступ к параметрам и градиентам
for p in model.parameters():
    assert p.grad.shape == p.data.shape

# обнуление градиентов перед следующей итерацией
model.zero_grad()
```

Оптимизаторы и тренировочный цикл появятся в следующих итерациях.

## Лицензия

MIT. См. [`LICENSE`](LICENSE).
