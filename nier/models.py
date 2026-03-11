from enum import Enum
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


class Activation(Enum):
    SIGMOID = "sigmoid"
    RELU    = "relu"
    LINEAR  = "linear"


@dataclass(frozen=True)
class Layer:
    size: int
    activation: Activation


@dataclass(frozen=True)
class Perceptron:
    layers:  tuple[Layer, ...]
    weights: tuple[NDArray, ...]


@dataclass(frozen=True)
class TrainingResult:
    perceptron:   Perceptron
    loss_history: tuple[float, ...]
