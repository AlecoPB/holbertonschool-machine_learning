#!/usr/bin/env python3
import numpy as np
"""
This module defines the Neuron class
"""


class Neuron:
    """
    This is the Neuron module
    The __init__ function initializes attributes
    and checks for errors
    """
    def __innit__(self, nx):
        if not isinstance(int, nx):
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")
        self.W = np.random.normal()
        self.b = 0
        self.A = 0
