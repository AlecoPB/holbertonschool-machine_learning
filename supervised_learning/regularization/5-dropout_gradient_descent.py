#!/usr/bin/env python3
"""_summary_
This is some documentation
"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights of a neural network
    with Dropout regularization using gradient descent

    Args:
        Y: numpy.ndarray - one-hot encoded correct labels of shape (classes, m)
        weights: dict - dictionary of weights and biases of the neural network
        cache: dict - dictionary containing
        the outputs of each layer and the dropout masks used
        alpha: float - learning rate
        keep_prob: float - probability that a node will be kept
        L: int - number of layers in the network

    Returns:
        None (updates weights in place)
    """
    m = Y.shape[1]
    dZ = cache['A' + str(L)] - Y

    for lay in range(L, 0, -1):
        dW = np.dot(dZ, cache['A' + str(lay - 1)].T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        if l > 1:
            dA_prev = np.dot(weights['W' + str(lay)].T, dZ)
            dA_prev = np.multiply(dA_prev, cache['D' + str(lay - 1)])
            dA_prev /= keep_prob
            dZ = dA_prev * (1 - np.power(cache['A' + str(lay - 1)], 2))
        weights['W' + str(lay)] -= alpha * dW
        weights['b' + str(lay)] -= alpha * db
