import numpy as np
from numpy.typing import NDArray


def min_max(data: NDArray) -> NDArray:
    min_val = data.min()
    max_val = data.max()

    if max_val == min_val:
        return np.zeros_like(data, dtype=float)

    return (data - min_val) / (max_val - min_val)
