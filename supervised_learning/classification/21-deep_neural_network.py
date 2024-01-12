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

    def evaluate(self, X, Y):
        """

        Args:
            X (np.array): Input set
            Y (np.array): Correct lables for the data

        Returns:
            np.array: Evaluated predictions
            np.array: Cost
        """
        self.forward_prop(X)

        cost = self.cost(Y, self.__cache['A' + str(self.__L)])
        A = np.where(self.__cache['A' + str(self.__L)] >= 0.5, 1, 0)
        return A, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Args:
            X (np.array): Input set
            Y (np.array): Correct labels for data
            A1 (np.array): Activated outputs of the hidden layer
            A2 (np.array): Activated outputs of the output layer
            alpha (float, optional): Neuron learning rate. Defaults to 0.05.
        """
        m = Y.shape[1]
        for i in reversed(range(self.L)):
            if i == self.L - 1:
                dz = cache['A' + str(i + 1)] - Y
            else:
                dz = np.matmul(self.weights['W' + str(i + 1)].T, dz_prev) * (cache['A' + str(i + 1)] * (1 - cache['A' + str(i + 1)]))
            dw = np.matmul(dz, cache['A' + str(i)].T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m
            dz_prev = dz
            self.weights['W' + str(i + 1)] -= alpha * dw
            self.weights['b' + str(i + 1)] -= alpha * db
