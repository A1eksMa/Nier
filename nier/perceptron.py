from typing import List

import numpy as np
from numpy.typing import NDArray

from .activation import apply
from .models import Layer, Perceptron
from .validate import validate_layers


def create(layers: List[Layer]) -> Perceptron:
    validate_layers(layers)

    weights = tuple(
        np.random.uniform(-1.0, 1.0, size=(layers[i].size, layers[i + 1].size))
        for i in range(len(layers) - 1)
    )

    return Perceptron(layers=tuple(layers), weights=weights)


def feed_forward(net: Perceptron, inputs: List[float]) -> tuple[NDArray, ...]:
    return tuple(_forward(net.layers, list(net.weights), inputs))


# internal helper — shared with training.py
def _forward(
    layers:  tuple[Layer, ...],
    weights: List[NDArray],
    inputs:  List[float],
) -> List[NDArray]:
    values: List[NDArray] = [np.array(inputs, dtype=float)]

    for i, weight in enumerate(weights):
        z = np.dot(weight.T, values[i])
        values.append(apply(z, layers[i + 1].activation))

    return values
