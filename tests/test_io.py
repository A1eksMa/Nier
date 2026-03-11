import json
import numpy as np
import pytest

import nier
from nier.models import Activation, Layer, Perceptron
from nier.io import save, load


@pytest.fixture
def net():
    np.random.seed(42)
    return nier.create([
        Layer(2, Activation.SIGMOID),
        Layer(4, Activation.RELU),
        Layer(1, Activation.LINEAR),
    ])


def test_save_creates_file(net, tmp_path):
    path = tmp_path / "model.json"
    save(net, path)
    assert path.exists()


def test_saved_file_is_valid_json(net, tmp_path):
    path = tmp_path / "model.json"
    save(net, path)
    data = json.loads(path.read_text())
    assert "layers" in data
    assert "weights" in data


def test_roundtrip_architecture(net, tmp_path):
    path = tmp_path / "model.json"
    save(net, path)
    loaded = load(path)
    assert loaded.layers == net.layers


def test_roundtrip_weights(net, tmp_path):
    path = tmp_path / "model.json"
    save(net, path)
    loaded = load(path)
    for original, restored in zip(net.weights, loaded.weights):
        np.testing.assert_array_almost_equal(original, restored)


def test_roundtrip_predictions(net, tmp_path):
    path = tmp_path / "model.json"
    save(net, path)
    loaded = load(path)
    inputs = [0.5, 0.8]
    original_out = nier.feed_forward(net, inputs)[-1]
    loaded_out   = nier.feed_forward(loaded, inputs)[-1]
    np.testing.assert_array_almost_equal(original_out, loaded_out)


def test_load_returns_perceptron(net, tmp_path):
    path = tmp_path / "model.json"
    save(net, path)
    loaded = load(path)
    assert isinstance(loaded, Perceptron)


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load(tmp_path / "nonexistent.json")


def test_save_load_via_nier_namespace(net, tmp_path):
    path = tmp_path / "model.json"
    nier.save(net, path)
    loaded = nier.load(path)
    assert loaded.layers == net.layers
