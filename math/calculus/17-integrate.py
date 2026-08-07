#!/usr/bin/env python3
"""Define function poly_integral."""


def poly_integral(poly, C=0):
    """Calculate the integral of a polynomial."""
    if not isinstance(poly, list)\
            or not poly\
            or any(not isinstance(elem, int) for elem in poly)\
            or not isinstance(C, int):
        return None
    deriv = [C]
    for i, v in enumerate(poly):
        int_div, remainder = divmod(v, i + 1)
        # divmod(num, den) = (num // den, num % den)
        deriv.append(int_div if remainder == 0 else v / (i + 1))
    return deriv
