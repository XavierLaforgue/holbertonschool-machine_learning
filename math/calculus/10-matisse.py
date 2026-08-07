#!/usr/bin/env python3
"""Define function poly_derivative."""
import numpy as np


def poly_derivative(poly):
    """Calculate the derivative of a polynomial."""
    if not isinstance(poly, list)\
            or not poly\
            or any(not isinstance(elem, int) for elem in poly):
        return None
    if len(poly) == 1:
        return [0]
    deriv = list(np.asarray(poly[1:]) * np.arange(1, len(poly)))
    # deriv = []
    # for idx, val in enumerate(poly):
    #     if idx == 0:
    #         continue
    #     deriv.append(idx * val)
    return deriv
