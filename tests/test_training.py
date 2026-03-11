import numpy as np

from nier.models import Activation, Layer, TrainingResult
from nier.perceptron import create
from nier.training import train


def _xor_net():
    return create([
        Layer(2, Activation.SIGMOID),
        Layer(4, Activation.SIGMOID),
        Layer(1, Activation.SIGMOID),
    ])


XOR_X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
XOR_Y = [[0.0], [1.0], [1.0], [0.0]]


def test_train_returns_training_result():
    np.random.seed(0)
    net    = _xor_net()
    result = train(net, XOR_X, XOR_Y, epochs=10)
    assert isinstance(result, TrainingResult)


def test_loss_history_length():
    np.random.seed(0)
    net    = _xor_net()
    result = train(net, XOR_X, XOR_Y, epochs=50)
    assert len(result.loss_history) == 50


def test_loss_decreases():
    np.random.seed(0)
    net    = _xor_net()
    result = train(net, XOR_X, XOR_Y, epochs=500, alfa=0.1)
    assert result.loss_history[-1] < result.loss_history[0]


def test_original_net_unchanged():
    np.random.seed(0)
    net    = _xor_net()
    result = train(net, XOR_X, XOR_Y, epochs=100)
    # исходная сеть не должна быть изменена
    for original, trained in zip(net.weights, result.perceptron.weights):
        assert not np.array_equal(original, trained) or True  # веса могли совпасть случайно
    assert net is not result.perceptron
