#!/usr/bin/env python3
"""Define function add_matrices."""


def add_matrices(mat1, mat2):
    """Add the two input matrices."""
    def get_shape(mat: list) -> list[int]:
        shape = []
        while isinstance(mat, list):
            shape.append(len(mat))
            mat = mat[0]
        return shape
    mat1_shape = shape = get_shape(mat1)
    mat2_shape = get_shape(mat2)
    if mat1_shape != mat2_shape:
        return None
    if len(shape) == 1:
        res = [mat1[i] + mat2[i] for i in range(shape[0])]
        return res
    res = []
    for i in range(shape[0]):
        res.append(add_matrices(mat1[i], mat2[i]))
    return res
