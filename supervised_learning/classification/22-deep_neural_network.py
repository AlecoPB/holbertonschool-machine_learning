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
        Calculate one pass of gradient descent on the neural network

        Args:
            Y (np.array): Correct labels for the data
            cache (dict): Activated outputs of each layer
            alpha (float): Learning rate

        Returns:
            dict: Updated weights and biases
        """
        m = Y.shape[1]
        weights_copy = self.__weights.copy()
        dz = cache['A' + str(self.__L)] - Y

        for i in range(self.__L, 0, -1):
            dw = np.matmul(cache['A' + str(i - 1)], dz.T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m
            dz = np.matmul(weights_copy['W' + str(i)].T,
                           dz) * (cache['A' + str(i - 1)]
                                  * (1 - cache['A' + str(i - 1)]))

            self.__weights['W' + str(i)] -= alpha * dw.T
            self.__weights['b' + str(i)] -= alpha * db

        return self.__weights

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """_summary_

        Args:
            X (_type_): _description_
            Y (_type_): _description_
            iterations (int, optional): _description_. Defaults to 5000.
            alpha (float, optional): _description_. Defaults to 0.05.

        Raises:
            TypeError: _description_
            ValueError: _description_
            TypeError: _description_
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        if not type(iterations) is int:
            raise TypeError("iterations must be an integer")
        elif iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if not type(alpha) is float:
            raise TypeError("alpha must be a float")
        elif alpha < 0:
            raise ValueError("alpha must be positive")
        for i in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(Y, self.cache, alpha)
        return self.evaluate(X, Y)
