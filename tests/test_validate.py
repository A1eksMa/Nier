import pytest

from nier.models import Activation, Layer
from nier.validate import validate_layers


def test_valid_layers():
    layers = [Layer(3, Activation.SIGMOID), Layer(1, Activation.SIGMOID)]
    validate_layers(layers)  # должен пройти без исключений


def test_not_a_list():
    with pytest.raises(TypeError):
        validate_layers((Layer(3, Activation.SIGMOID), Layer(1, Activation.SIGMOID)))


def test_too_few_layers():
    with pytest.raises(ValueError):
        validate_layers([Layer(3, Activation.SIGMOID)])


def test_invalid_element():
    with pytest.raises(TypeError):
        validate_layers([Layer(3, Activation.SIGMOID), "not a layer"])


def test_zero_size_layer():
    with pytest.raises(ValueError):
        validate_layers([Layer(3, Activation.SIGMOID), Layer(0, Activation.SIGMOID)])
