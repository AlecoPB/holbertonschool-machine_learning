#!/usr/bin/env python3
"""
This module defines the Neuron class
"""
import numpy as np


class Neuron:
    """
    This is the Neuron module
    The __init__ function initializes attributes
    and checks for errors
    """
    def __init__(self, nx):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.normal(loc=0.0, scale=1.0, size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A

    def forward_prop(self, X):
        """
        Args:
            X is an array containing input sets

        Returns:
            The private attribute A as an array
        """
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

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
            Y (np.array): Correct labes for the data

        Returns:
            np.array: Evaluated predictions
            np.array: Cost
        """
        act_X = self.forward_prop(X)
        cost = self.cost(Y, act_X)
        act_X = (act_X > 0.5).astype(int)
        return act_X, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Args:
            X (np.array): Input set
            Y (np.array): Correct labels for data
            A (np.array): Activated outputs
            alpha (float, optional): Neuron learning rate. Defaults to 0.05.
        """
        # cst = self.cost(Y, A)
        error = A - Y
        m = np.shape(X)[1]
        dw = np.dot(X, error.T) / m
        db = np.sum(error) / m
        self.__W = self.__W - alpha * dw
        self.__b = self.__b - alpha * db
        # f_cst = self.cost(Y, self.forward_prop(X))
