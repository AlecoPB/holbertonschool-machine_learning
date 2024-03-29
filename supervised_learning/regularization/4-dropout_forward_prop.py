#!/usr/bin/env python3
"""_summary_

Returns:
    _type_: _description_
"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Dropout

    Args:
        X: numpy.ndarray - input data for the network of shape (nx, m)
        weights: dict - dictionary of weights and biases of the neural network
        L: int - number of layers in the network
        keep_prob: float - probability that a node will be kept

    Returns:
        dict: a dictionary containing the outputs
        of each layer and the dropout mask used on each layer
    """
    cache = {}
    cache['A0'] = X
    for lay in range(1, L + 1):
        Z = np.dot(weights['W' + str(lay)],
                   cache['A' + str(lay - 1)]) + weights['b' + str(lay)]
        if lay == L:
            A = np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True)
        else:
            A = np.tanh(Z)
            D = (np.random.rand(A.shape[0],
                                A.shape[1]) < keep_prob).astype(int)
            A = np.multiply(A, D)
            A /= keep_prob
            cache['D' + str(lay)] = D
        cache['A' + str(lay)] = A
    return cache
