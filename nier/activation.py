import numpy as np
from numpy.typing import NDArray

from .models import Activation


def apply(x: NDArray, activation: Activation) -> NDArray:
    if activation == Activation.SIGMOID:
        return 1.0 / (1.0 + np.exp(-x))

    if activation == Activation.RELU:
        return np.maximum(0.0, x)

    if activation == Activation.LINEAR:
        return x

    raise ValueError(f"unknown activation: {activation}")


def derivative(x: NDArray, activation: Activation) -> NDArray:
    if activation == Activation.SIGMOID:
        s = apply(x, activation)
        return s * (1.0 - s)

    if activation == Activation.RELU:
        return np.where(x > 0.0, 1.0, 0.0)

    if activation == Activation.LINEAR:
        return np.ones_like(x)

    raise ValueError(f"unknown activation: {activation}")
