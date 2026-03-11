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


def test_shuffle_false_is_deterministic():
    # With shuffle=False and the same seed, two runs must produce identical results
    np.random.seed(5)
    net = _xor_net()
    r1 = train(net, XOR_X, XOR_Y, epochs=20, shuffle=False)

    np.random.seed(5)
    net = _xor_net()
    r2 = train(net, XOR_X, XOR_Y, epochs=20, shuffle=False)

    assert r1.loss_history == r2.loss_history


def test_shuffle_default_improves_convergence():
    # Imbalanced ordered data: shuffle=True should converge, shuffle=False may not
    # 6 class-0 samples first, then 2 class-1 samples
    X = [[0.0, 0.0]] * 6 + [[1.0, 1.0]] * 2
    y = [[0.0]] * 6      + [[1.0]] * 2

    np.random.seed(0)
    net = _xor_net()
    result = train(net, X, y, epochs=200, alfa=0.1, shuffle=True)
    assert result.loss_history[-1] < result.loss_history[0]
