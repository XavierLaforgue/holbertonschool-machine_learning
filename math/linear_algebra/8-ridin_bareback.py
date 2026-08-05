#!/usr/bin/env python3
"""Define function mat_mul"""


def mat_mul(mat1, mat2):
    """Perform matrix multiplication"""
    mat1_shape = (len(mat1), len(mat1[0]))
    mat2_shape = (len(mat2), len(mat2[0]))
    if mat1_shape[1] != mat2_shape[0]:
        return None
    res = []
    for i in range(mat1_shape[0]):
        res.append([])
        for j in range(mat2_shape[1]):
            elem = sum(mat1[i][k] * mat2[k][j] for k in range(mat1_shape[1]))
            res[i].append(elem)
    return res
