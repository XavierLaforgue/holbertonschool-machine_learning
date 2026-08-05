#!/usr/bin/env python3
"""Define function cat_matrices2D"""


def cat_matrices2D(mat1, mat2, axis=0):
    """Concatenate two matrices along a specific axis"""
    mat1_shape = (len(mat1), len(mat1[0]))
    mat2_shape = (len(mat2), len(mat2[0]))
    if axis == 0:
        if mat1_shape[1] != mat2_shape[1]:
            return None
        return mat1 + mat2
    if axis == 1:
        if mat1_shape[0] != mat2_shape[0]:
            return None
        res = []
        for i in range(mat1_shape[0]):
            res.append(mat1[i] + mat2[i])
        return res
