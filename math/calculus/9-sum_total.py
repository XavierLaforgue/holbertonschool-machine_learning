#!/usr/bin/env python3
"""Define function summation_i_squared."""
import numpy as np


def summation_i_squared(n):
    r"""Calculate \\sigma_{i=1}^n i^2."""
    if not isinstance(n, int) or n < 1:
        return None
    return np.sum(np.arange(1, n + 1, 1)**2)
