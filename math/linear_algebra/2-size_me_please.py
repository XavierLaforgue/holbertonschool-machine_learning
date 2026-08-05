#!/usr/bin/env python3
"""Define function matrix_shape"""


def matrix_shape(matrix: list) -> list[int]:
    """Calculate the shape of the input matrix"""
    shape = [len(matrix)]
    if isinstance(matrix[0], int):
        return shape
    shape.extend(matrix_shape(matrix[0]))
    return shape
