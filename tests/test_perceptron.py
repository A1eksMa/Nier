import pytest
import numpy as np

from nier.models import Activation, Layer
from nier.perceptron import create, feed_forward


LAYERS = [
    Layer(2, Activation.SIGMOID),
    Layer(3, Activation.SIGMOID),
    Layer(1, Activation.SIGMOID),
]


def test_create_returns_perceptron():
    from nier.models import Perceptron
    net = create(LAYERS)
    assert isinstance(net, Perceptron)


def test_create_weights_shape():
    net = create(LAYERS)
    assert net.weights[0].shape == (2, 3)
    assert net.weights[1].shape == (3, 1)


def test_create_raises_on_invalid_layers():
    with pytest.raises(ValueError):
        create([Layer(2, Activation.SIGMOID)])


def test_feed_forward_output_length():
    net = create(LAYERS)
    values = feed_forward(net, [0.5, 0.8])
    assert len(values) == len(LAYERS)


def test_feed_forward_output_range():
    net = create(LAYERS)
    values = feed_forward(net, [0.5, 0.8])
    output = values[-1]
    assert np.all(output > 0) and np.all(output < 1)
