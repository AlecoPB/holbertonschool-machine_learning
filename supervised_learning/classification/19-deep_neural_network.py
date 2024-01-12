#!/usr/bin/env python3
"""
This module contains a class defining a Neural Network
"""
import numpy as np


class DeepNeuralNetwork:
    """
    A Neural Network class
    """
    def __init__(self, nx, layers):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        elif not all(map(lambda x: x > 0 and isinstance(x, int), layers)):
            raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for i in range(self.L):
            if i == 0:
                self.__weights['W' + str(i + 1)] =\
                    np.random.randn(layers[i], nx) * np.sqrt(2 / nx)
            else:
                self.__weights['W' + str(i + 1)] = np.random.randn(
                    layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1])
            self.__weights['b' + str(i + 1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    def forward_prop(self, X):
        """0

        Args:
            X (_type_): _description_

        Returns:
            _type_: _description_
        """
        self.__cache['A0'] = X
        for i in range(self.L):
            Z = np.matmul(self.weights['W' + str(i + 1)],
                          self.__cache['A' + str(i)])\
                              + self.weights['b' + str(i + 1)]
            self.__cache['A' + str(i + 1)] = 1.0 / (1.0 + np.exp(-Z))
        return self.__cache['A' + str(self.L)], self.__cache

    def cost(self, Y, A):
        """

        Args:
            Y (np.array): Correct labels for the data
            A (np.array): Activated outputs

        Returns:
            Cost
        """
        m = np.shape(Y)[1]
        sum = np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return (-1 / m) * sum