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
    def __init__(self, nx):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")
        self.W = np.random.normal(loc=0.0, scale=1.0, size=775)
        self.b = 0
        self.A = 0
