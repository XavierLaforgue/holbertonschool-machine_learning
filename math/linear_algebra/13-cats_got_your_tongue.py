#!/usr/bin/env python3
"""Define function np_cat"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """Concatenate the two input matrices along the specified axis"""
    return np.concatenate((mat1, mat2), axis=axis)
