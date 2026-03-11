import numpy as np
import pytest

from nier.activation import apply, derivative
from nier.models import Activation


def test_sigmoid_range():
    x = np.array([-10.0, 0.0, 10.0])
    result = apply(x, Activation.SIGMOID)
    assert np.all(result > 0) and np.all(result < 1)


def test_sigmoid_midpoint():
    assert apply(np.array([0.0]), Activation.SIGMOID)[0] == pytest.approx(0.5)


def test_relu_negative_zeroed():
    x = np.array([-3.0, -1.0, 0.0, 2.0])
    result = apply(x, Activation.RELU)
    assert np.all(result >= 0)
    assert result[0] == 0.0 and result[3] == 2.0


def test_linear_passthrough():
    x = np.array([1.5, -2.3, 0.0])
    np.testing.assert_array_equal(apply(x, Activation.LINEAR), x)


def test_sigmoid_derivative_shape():
    x = np.array([0.0, 1.0, -1.0])
    d = derivative(x, Activation.SIGMOID)
    assert d.shape == x.shape


def test_linear_derivative_ones():
    x = np.array([3.0, -1.0, 0.0])
    np.testing.assert_array_equal(derivative(x, Activation.LINEAR), np.ones_like(x))
