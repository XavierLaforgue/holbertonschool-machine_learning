#!/usr/bin/env python3
"""Define function add_arrays"""


def add_arrays(arr1, arr2):
    """Add two arrays"""
    if len(arr1) != len(arr2):
        return None
    res = [elem1 + elem2 for elem1, elem2 in zip(arr1, arr2)]
    return res
