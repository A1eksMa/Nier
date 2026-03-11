from typing import List

import numpy as np
from numpy.typing import NDArray

from .activation import derivative
from .models import Perceptron, TrainingResult
from .perceptron import _forward


def train(
    net:     Perceptron,
    X:       List[List[float]],
    y:       List[List[float]],
    epochs:  int   = 1000,
    alfa:    float = 0.01,
    shuffle: bool  = True,
) -> TrainingResult:
    # mutable copies of weights for the training loop
    weights: List[NDArray] = [w.copy() for w in net.weights]
    loss_history: List[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0

        if shuffle:
            idx    = np.random.permutation(len(X))
            X_iter = [X[i] for i in idx]
            y_iter = [y[i] for i in idx]
        else:
            X_iter, y_iter = X, y

        for xi, yi in zip(X_iter, y_iter):
            values = _forward(net.layers, weights, xi)
            weights, sample_loss = _backprop(net, weights, values, np.array(yi, dtype=float), alfa)
            epoch_loss += sample_loss

        loss_history.append(epoch_loss / len(X))

    trained = Perceptron(
        layers=net.layers,
        weights=tuple(w.copy() for w in weights),
    )

    return TrainingResult(
        perceptron=trained,
        loss_history=tuple(loss_history),
    )


def _backprop(
    net:     Perceptron,
    weights: List[NDArray],
    values:  List[NDArray],
    target:  NDArray,
    alfa:    float,
) -> tuple[List[NDArray], float]:
    loss  = float(np.mean((target - values[-1]) ** 2))
    error = target - values[-1]

    new_weights = [w.copy() for w in weights]

    for j in range(len(new_weights) - 1, -1, -1):
        activation = net.layers[j + 1].activation
        gradient   = error * derivative(values[j + 1], activation)
        delta      = new_weights[j] @ gradient
        new_weights[j] += alfa * np.outer(values[j], gradient)
        error = delta

    return new_weights, loss
