#!/usr/bin/env python3
"""Define the `Neuron` class."""
import numpy as np


class Neuron:
    """Define single neuron performing binary classification."""

    def __init__(self, nx):
        """Initialize neuron.

        nx: number of inpyt features to the neuron.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(1, nx)
        # in randn(d0, d1, ...): d0, d1, are the dimensions of the returned
        # array
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter for weights vector."""
        return self.__W

    @property
    def b(self):
        """Getter for bias."""
        return self.__b

    @property
    def A(self):
        """Getter for activated output."""
        return self.__A
