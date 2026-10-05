# numpy-neural-networks

Учебный проект по глубокому обучению. Состоит из двух частей:

1. **`numpy_nn`** — библиотека нейросетей, реализованная с нуля на NumPy.
   Цель — понять каждую формулу и каждый алгоритм через собственную
   реализацию, без использования фреймворков. **Проект заморожен:**
   основные концепции (слои, активации, лоссы, оптимизаторы) реализованы
   и покрыты тестами.
2. **PyTorch-практикум** — активная часть. Изучение современных тем
   (регуляризация, нормализация, свёртки, рекуррентные сети, обучение
   на GPU) ведётся на PyTorch. К `numpy_nn` возвращаюсь точечно, когда
   нужно разобрать конкретный алгоритм изнутри.

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

## Что реализовано в `numpy_nn`

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

## Прогресс по темам

### Блок 1. Базовые концепции (закрыт)

1. [x] Модель нейрона МакКаллока–Питтса (удалён из библиотеки, см. [`reports/01_mp_neuron.md`](reports/01_mp_neuron.md))
2. [x] Строение многослойного перцептрона
3. [x] Функции активации
4. [x] Прямой и обратный проход
5. [x] Обучение нейронной сети
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
16. [x] Сравнение оптимизаторов ([`reports/compare_optimizers.md`](reports/compare_optimizers.md))

### Блок 2. Обучение и регуляризация (в работе)

- [ ] Переобучение и недообучение
- [ ] Методы борьбы с недообучением
- [ ] Методы борьбы с переобучением
- [ ] Label Smoothing
- [ ] Инициализация весов модели
- [ ] Предобучение, дообучение и Fine-tuning
- [ ] Warm-up и Schedulers
- [ ] Гиперпараметры
- [ ] Data Drift
- [ ] Concept Drift

### Блок 3. Архитектурные компоненты

- [ ] Batch Normalization
- [ ] DropOut
- [ ] Мультиколлинеарность

### Блок 4. Продвинутые темы

- [ ] Свёртка
- [ ] Pooling
- [ ] Взрыв и затухание градиентов
- [ ] Остаточные связи
- [ ] Аугментация данных
- [ ] RNN: Vanilla, LSTM и GRU
- [ ] Обучение на CPU и GPU
- [ ] Обучение нейросети на одной GPU: полный цикл
- [ ] Обучение на нескольких GPU: DataParallel, DDP, torchrun
- [ ] Виды параллелизма: data, tensor, pipeline, FSDP
- [ ] Экономия видеопамяти: AMP, checkpointing, accumulation

## Структура проекта

```
src/numpy_nn/
├── nn/                 # слои, активации, контейнеры, функции потерь
│   ├── core.py         # Parameter, Module
│   ├── layers.py       # Linear
│   ├── activations.py  # функции и модули активаций
│   ├── sequential.py   # Sequential
│   └── losses.py       # CrossEntropyLoss, MSELoss
├── optim/              # оптимизаторы
│   ├── base.py         # Optimizer — базовый интерфейс
│   ├── sgd.py
│   ├── momentum.py
│   ├── nesterov.py
│   ├── adagrad.py
│   ├── rmsprop.py
│   ├── adam.py
│   ├── adamw.py
│   └── nadam.py
└── utils/              # утилиты
    └── ema.py          # экспоненциальное скользящее среднее

tests/                  # pytest-тесты, по одному файлу на модуль
examples/               # примеры и скрипты-эксперименты
notebooks/              # jupyter-ноутбуки (PyTorch-практикум)
reports/                # текстовые отчёты и результаты экспериментов
figures/                # графики (не версионируются)
data/                   # локальные датасеты (не версионируются)
```

## Использование `numpy_nn`

Пример: MLP для многоклассовой классификации.

```python
import numpy as np
from numpy_nn.nn import CrossEntropyLoss, Linear, ReLU, Sequential
from numpy_nn.optim import Adam

model = Sequential(
    Linear(784, 128),
    ReLU(),
    Linear(128, 64),
    ReLU(),
    Linear(64, 10),
)

criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-3)

# один шаг обучения
optimizer.zero_grad()
logits = model(x)
loss = criterion(logits, y)
dlogits = criterion.backward()
model.backward(dlogits)
optimizer.step()
```

Полный пример со сравнением оптимизаторов на MNIST —
[`examples/compare_optimizers.py`](examples/compare_optimizers.py).

## Лицензия

MIT. См. [`LICENSE`](LICENSE).
