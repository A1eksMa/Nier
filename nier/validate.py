from .models import Layer


def validate_layers(layers: list[Layer]) -> None:
    if not isinstance(layers, list):
        raise TypeError(f"layers must be a list, got {type(layers).__name__}")

    if len(layers) < 2:
        raise ValueError(f"layers must have at least 2 elements, got {len(layers)}")

    for i, layer in enumerate(layers):
        if not isinstance(layer, Layer):
            raise TypeError(f"element at index {i} must be a Layer, got {type(layer).__name__}")

        if not isinstance(layer.size, int) or layer.size <= 0:
            raise ValueError(f"layer size must be a positive integer, got {layer.size!r} at index {i}")
