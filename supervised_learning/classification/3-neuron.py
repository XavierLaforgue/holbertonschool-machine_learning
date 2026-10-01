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

    def forward_prop(self, X):
        """Calculate forward propagation of the neuron.

        X: numpy.ndarray with shape `(nx, m)` containing input data.
        nx: number of input features to the neuron.
        m: number of examples.
        Return: activated output `A`.
        """
        # neuron eq: z = w_1 * x_1 + w_2 * x_2 + ... + b
        Z = np.matmul(self.__W, X) + self.__b

        def sigmoid(weighted_sum):
            """Define sigmoid function."""
            return 1 / (1 + np.exp(-weighted_sum))
        self.__A = sigmoid(Z)
        return self.__A

    def cost(self, Y, A):
        """Calculate cost of the model using logistic regression.

        Y: numpy.ndarray with shape `(1, m)` that contains the correct labels
            for input data.
        A: numpy.ndarray with shape `(1, m)` activated ouput of the neuron for
            each example.
        Return cost."""
        # logistic regression eq:
        loss = Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        m = Y.shape[1]
        cost = -np.sum(loss) / m
        return cost
