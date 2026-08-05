#!/usr/bin/env python3
"""Define the function matrix_transpose"""


def matrix_transpose(matrix: list[list[int]]) -> list[list[int]]:
    """Return the transpose of a 2D matrix"""
    n_rows = len(matrix)
    n_cols = len(matrix[0])
    transposed = [[] for _ in range(n_cols)]
    for i in range(n_rows):
        for j in range(n_cols):
            transposed[j].append(matrix[i][j])
    return transposed
