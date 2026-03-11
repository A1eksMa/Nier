# Nier

A minimal neural network library in pure Python / NumPy — multilayer perceptron with backpropagation.

## Features

- Immutable `Perceptron` and `Layer` dataclasses
- Activation functions: Sigmoid, ReLU, Linear
- Mini-batch-free backpropagation (online learning)
- Min-max normalization helper
- Dockerized environment

## Installation

```bash
pip install -r requirements.txt
```

## Quick start

```python
import nier
from nier.models import Activation, Layer

# Define architecture
layers = [
    Layer(size=2, activation=Activation.LINEAR),
    Layer(size=4, activation=Activation.SIGMOID),
    Layer(size=1, activation=Activation.SIGMOID),
]

# Create network
net = nier.create(layers)

# Train
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [[0],     [1],    [1],    [0]]

result = nier.train(net, X, y, epochs=5000, alfa=0.1)

# Predict
outputs = nier.feed_forward(result.perceptron, [1, 0])
print(outputs[-1])  # final layer activations
```

## API

| Function | Description |
|---|---|
| `nier.create(layers)` | Create a new perceptron with random weights |
| `nier.feed_forward(net, inputs)` | Run forward pass, return all layer activations |
| `nier.train(net, X, y, epochs, alfa)` | Train with backpropagation, return `TrainingResult` |
| `nier.min_max(data)` | Min-max normalize a list of values |

## Running tests

```bash
pytest
```

## Docker

```bash
docker build -t nier .
docker run --rm nier
```

## License

[MIT](LICENSE)
