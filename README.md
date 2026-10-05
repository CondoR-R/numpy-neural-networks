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

- [x] SGD
- [x] Momentum
- [x] Nesterov Accelerated Momentum (NAG)
- [x] AdaGrad
- [x] RMSProp
- [x] Adam
- [x] AdamW
- [x] Nadam

### Утилиты

- [x] Экспоненциальное скользящее среднее (EMA)
- [ ] Learning rate schedulers

## Прогресс по темам

1. [x] Модель нейрона МакКаллока–Питтса (удалён из библиотеки, см. [`reports/`](reports/01_mp_neuron.md))
2. [x] Строение многослойного перцептрона
3. [x] Функции активации
4. [x] Прямой и обратный проход
5. [x] Обучение нейронной сети (частично: forward/backward есть, тренировочный цикл — нет)
6. [x] Оптимизаторы в глубоком обучении
7. [x] Экспоненциальное скользящее среднее
8. [x] SGD
9. [x] Momentum
10. [x] NAG
11. [x] AdaGrad
12. [x] RMSProp
13. [x] Adam
14. [x] AdamW
15. [x] Nadam
16. [x] Сравнение оптимизаторов

## Структура проекта

```
src/numpy_nn/
├── nn/ # всё, что связано с моделью и обучением
│ ├── core.py # Parameter, Module — базовые абстракции
│ ├── layers.py # Linear
│ ├── activations.py # функции и модули активаций (ReLU, Sigmoid, Tanh)
│ ├── sequential.py # Sequential — контейнер для композиции слоёв
│ └── losses.py # CrossEntropyLoss, MSELoss
├── optim/ # оптимизаторы
│ ├── base.py # Optimizer — базовый интерфейс
│ ├── sgd.py # SGD
│ ├── momentum.py # Momentum
│ ├── nesterov.py # Nesterov Accelerated Gradient
│ ├── adagrad.py # AdaGrad
│ ├── rmsprop.py # RMSProp
│ ├── adam.py # Adam
│ ├── adamw.py # AdamW
│ └── nadam.py # Nadam
├── utils/ # утилиты
│ └── ema.py # экспоненциальное скользящее среднее
└── data/ # загрузчики датасетов (пусто)

tests/ # pytest-тесты, по одному файлу на модуль
examples/ # примеры использования
notebooks/ # эксперименты и конспекты
reports/ # текстовые отчёты по темам
figures/ # графики (не версионируются)
data/ # локальные датасеты (не версионируются)
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
