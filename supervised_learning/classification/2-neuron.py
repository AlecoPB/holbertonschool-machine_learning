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
        for n_ex in range(len(X)):
            sum = 0
            for i in range(nx):
                sum += self.__W[i, 0] * X[i, n_ex]
            sum += self.__b
            if n_ex == 1:
                self.__A = [sum]
            else:
                self.__A.append[sum]
            return self.__A
