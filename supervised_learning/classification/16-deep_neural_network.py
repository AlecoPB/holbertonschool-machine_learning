#!/usr/bin/env python3
"""
This module contains a class defining a Neural Network
"""
import numpy as np
# import matplotlib.pyplot as plt


class NeuralNetwork:
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
        elif not all(map(lambda x: x > 0 and type(x) is int, layers)):
            raise ValueError("nodes must be a positive integer")

        self.L = len(layers)
        cache = {}
        weights = {}
        for i in range(self.L):
            if i == 0:
                self.weights['W' + str(i + 1)] = np.random.randn(layers[i], nx) * np.sqrt(2 / nx)
            else:
                self.weights['W' + str(i + 1)] = np.random.randn(
                    layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1])
            self.weights['b' + str(i + 1)] = np.zeros((layers[i], 1))