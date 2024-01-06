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

        #Attributes
        W1 = np.random.normal(loc=0.0, scale=1.0, size=(1, nx)) #Weights vector for the hidden layer
        b1 = [0 for _ in range(nodes)] #Bias for the hidden layer
        A1 = 0 #Activated output for the hidden layer
        W2 = np.random.normal(loc=0.0, scale=1.0, size=(1, nx)) #Weights vector for the output neuron
        b2 = 0 #Bias for the output neuron
        A2 = 0 #Activated output for the output neuron (prediction)
