#!/usr/bin/env python3
"""Define function np_slice."""


def np_slice(matrix, axes=None):
    """Slice the input matrix along a specific axes."""
    if axes is None:
        return matrix
    if not axes:
        return matrix
    smart_idx = [slice(None)] * matrix.ndim
    for idx, sliz in axes.items():
        smart_idx[idx] = slice(*sliz)
    return matrix[tuple(smart_idx)]
