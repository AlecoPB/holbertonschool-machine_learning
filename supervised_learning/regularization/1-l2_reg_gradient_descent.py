#!/usr/bin/env python3
"""_summary_
This is some documentation
"""
import numpy as np


def softmax(Z):
    """Compute softmax values for each sets of scores in Z."""
    e_Z = np.exp(Z - np.max(Z))
    return e_Z / e_Z.sum(axis=0)

def tanh_derivative(x):
    """Compute the derivative of tanh function."""
    return 1 - np.tanh(x)**2

def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a neural network using gradient descent with L2 regularization.

    Parameters:
    Y: a one-hot numpy.ndarray of shape (classes, m) that contains the correct labels for the data
    weights: a dictionary of the weights and biases of the neural network
    cache: a dictionary of the outputs of each layer of the neural network
    alpha: the learning rate
    lambtha: the L2 regularization parameter
    L: the number of layers of the network

    The neural network uses tanh activations on each layer except the last, which uses a softmax activation.
    The weights and biases of the network should be updated in place.
    """
    m = Y.shape[1]
    weights_copy = weights.copy()
    dz_prev = cache['A' + str(L)] - Y

    for l in range(L, 0, -1):
        A_prev = cache['A' + str(l-1)]
        dw = (1/m) * np.dot(dz_prev, A_prev.T) + ((lambtha/m) * weights_copy['W' + str(l)])
        db = (1/m) * np.sum(dz_prev, axis=1, keepdims=True)
        if l > 1:
            dz_prev = np.dot(weights_copy['W' + str(l)].T, dz_prev) * tanh_derivative(cache['Z' + str(l-1)])

        weights['W' + str(l)] -= alpha * dw
        weights['b' + str(l)] -= alpha * db
