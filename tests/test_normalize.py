import numpy as np

from nier.normalize import min_max


def test_output_range():
    data   = np.array([10.0, 20.0, 30.0, 40.0])
    result = min_max(data)
    assert result.min() == 0.0
    assert result.max() == 1.0


def test_constant_array_returns_zeros():
    data   = np.array([5.0, 5.0, 5.0])
    result = min_max(data)
    np.testing.assert_array_equal(result, np.zeros(3))


def test_shape_preserved():
    data   = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = min_max(data)
    assert result.shape == data.shape
