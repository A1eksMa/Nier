import json
from pathlib import Path

import numpy as np

from .models import Activation, Layer, Perceptron


def save(net: Perceptron, path: str | Path) -> None:
    """Serialize a Perceptron to a JSON file."""
    data = {
        "layers": [
            {"size": layer.size, "activation": layer.activation.value}
            for layer in net.layers
        ],
        "weights": [w.tolist() for w in net.weights],
    }
    Path(path).write_text(json.dumps(data, indent=2))


def load(path: str | Path) -> Perceptron:
    """Deserialize a Perceptron from a JSON file."""
    data = json.loads(Path(path).read_text())

    layers = tuple(
        Layer(size=entry["size"], activation=Activation(entry["activation"]))
        for entry in data["layers"]
    )
    weights = tuple(np.array(w) for w in data["weights"])

    return Perceptron(layers=layers, weights=weights)
