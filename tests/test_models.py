import pytest
from dataclasses import FrozenInstanceError

from nier.models import Activation, Layer, Perceptron, TrainingResult
import numpy as np


def test_activation_values():
    assert Activation.SIGMOID.value == "sigmoid"
    assert Activation.RELU.value    == "relu"
    assert Activation.LINEAR.value  == "linear"


def test_layer_is_frozen():
    layer = Layer(size=4, activation=Activation.SIGMOID)
    with pytest.raises(FrozenInstanceError):
        layer.size = 8


def test_perceptron_is_frozen():
    layers  = (Layer(2, Activation.SIGMOID), Layer(1, Activation.SIGMOID))
    weights = (np.zeros((2, 1)),)
    net = Perceptron(layers=layers, weights=weights)
    with pytest.raises(FrozenInstanceError):
        net.layers = ()
