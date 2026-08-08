#!/usr/bin/env python3
"""Define function np_slice."""
import numpy as np


def np_slice(matrix: np.ndarray,
             axes: dict[int, tuple] | None = None) -> np.ndarray:
    """Slice the input matrix along a specific axes."""
    if axes is None:
        return matrix
    if not axes:
        return matrix
    smart_idx = [slice(None)] * matrix.ndim
    for idx, sliz in axes.items():
        smart_idx[idx] = slice(*sliz)
    return matrix[tuple(smart_idx)]
