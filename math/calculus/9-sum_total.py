#!/usr/bin/env python3
"""Define function summation_i_squared."""


def summation_i_squared(n):
    if not isinstance(n, int): return None
    return sum(i**2 for i in range(1, n+1))
