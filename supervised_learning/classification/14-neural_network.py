#!/usr/bin/env python3
"""
This module contains a class defining a Neural Network
"""
import numpy as np


class NeuralNetwork:
    """
    A Neural Network class
    """
    def __init__(self, nx, nodes):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        elif nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.normal(loc=0.0, scale=1.0, size=(nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(loc=0.0, scale=1.0, size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        return self.__W1

    @property
    def b1(self):
        return self.__b1

    @property
    def A1(self):
        return self.__A1

    @property
    def W2(self):
        return self.__W2

    @property
    def b2(self):
        return self.__b2

    @property
    def A2(self):
        return self.__A2

    def forward_prop(self, X):
        """
        Calculates the forward propagration of a neuron
        """
        self.__A1 = 1 / (1 + np.exp(-(np.dot(self.W1, X) + self.b1)))
        self.__A2 = 1 / (1 + np.exp(-(np.dot(self.W2, self.__A1) + self.b2)))

        return self.__A1, self.__A2

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
        self.__A1, self.__A2 = self.forward_prop(X)
        cost = self.cost(Y, self.__A2)
        A2 = np.where(self.__A2 >= 0.5, 1, 0)
        return A2, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Args:
            X (np.array): Input set
            Y (np.array): Correct labels for data
            A1 (np.array): Activated outputs of the hidden layer
            A2 (np.array): Activated outputs of the output layer
            alpha (float, optional): Neuron learning rate. Defaults to 0.05.
        """
        m = np.shape(X)[1]

        # Output layer error
        error2 = A2 - Y
        dw2 = np.dot(A1, error2.T) / m
        db2 = np.sum(error2, axis=1, keepdims=True) / m

        # Hidden layer error
        error1 = np.dot(self.__W2.T, error2) * A1 * (1 - A1)
        dw1 = np.dot(X, error1.T) / m
        db1 = np.sum(error1, axis=1, keepdims=True) / m

        # Update weights and biases
        self.__W1 -= alpha * dw1.T
        self.__b1 -= alpha * db1
        self.__W2 -= alpha * dw2.T
        self.__b2 -= alpha * db2

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Args:
            X (np_array)
            Y (np_array)
            iterations (int, optional): Times to iterate. Defaults to 5000.
            # alpha (float, optional): Learning rate. Defaults to 0.05.
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        elif iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        elif alpha < 0:
            raise ValueError("alpha must be positive")
        for n_itr in range(iterations):
            A1, A2 = self.forward_prop(X)
            self.gradient_descent(X, Y, A1, A2, alpha)
        return self.evaluate(X, Y)