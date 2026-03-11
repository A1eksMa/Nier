from typing import List

import numpy as np
from numpy.typing import NDArray

from .activation import apply
from .models import Layer, Perceptron
from .validate import validate_layers


def _init_weights(fan_in: int, fan_out: int, activation: "Activation") -> NDArray:
    from .models import Activation
    if activation == Activation.RELU:
        # He initialization: keeps variance stable through ReLU layers
        std = np.sqrt(2.0 / fan_in)
        return np.random.normal(0.0, std, size=(fan_in, fan_out))
    else:
        # Xavier / Glorot uniform: keeps variance stable for Sigmoid and Linear
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, size=(fan_in, fan_out))


def create(layers: List[Layer]) -> Perceptron:
    validate_layers(layers)

    weights = tuple(
        _init_weights(layers[i].size, layers[i + 1].size, layers[i + 1].activation)
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
