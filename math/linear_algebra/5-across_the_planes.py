#!/usr/bin/env python3
"""Define function add_matrices2D"""


def add_matrices2D(mat1, mat2):
    """Add two matrices element-wise"""
    mat1_shape = (len(mat1), len(mat1[0]))
    mat2_shape = (len(mat2), len(mat2[0]))
    if mat1_shape != mat2_shape:
        return None
    res = []
    for row1, row2 in zip(mat1, mat2):
        row = []
        for elem1, elem2 in zip(row1, row2):
            row.append(elem1 + elem2)
        res.append(row)
    return res
