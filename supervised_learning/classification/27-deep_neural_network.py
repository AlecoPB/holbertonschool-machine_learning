#!/usr/bin/env python3
"""
This module contains a class defining a Neural Network
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
one_hot_decode = __import__('one_hot_decode').one_hot_decode


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

    def sigmoid(self, Z):
        """
        Sigmoid activation function
        Args:
            Z (float)
        """
        return 1 / (1 + np.exp(-Z))

    def softmax(self, Z):
        """
        Softmax activation function
        """
        return np.exp(Z) / np.sum(np.exp(Z), axis=0)

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
            if i == self.L - 1:
                self.__cache['A' + str(i + 1)] = self.softmax(Z)
            else:
                self.__cache['A' + str(i + 1)] = self.sigmoid(Z)

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
        return -1 / m * np.sum(Y * np.log(A))

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
        A = one_hot_decode(self.__cache['A' + str(self.__L)], axis=0)
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

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
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
        costs = []
        for i in range(iterations + 1):
            self.forward_prop(X)
            cost = self.cost(Y, self.cache['A' + str(self.L)])
            if i % step == 0 or i == iterations:
                costs.append(cost)
                if verbose:
                    print("Cost after {} iterations: {}".format(i, cost))
            if i < iterations:
                self.gradient_descent(Y, self.cache, alpha)
        if graph:
            plt.plot(np.arange(0, iterations + 1, step), costs)
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()
        return self.evaluate(X, Y)

    def save(self, filename):
        """
        Saves the instance object to a file in pickle format
        """
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """
        Loads a pickled DeepNeuralNetwork object
        """
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None
