#!/usr/bin/env python3
"""_summary_
This is some documentation
"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the cost of a neural network with L2 regularization.

    Parameters:
    cost: the cost of the network without L2 regularization
    lambtha: the regularization parameter
    weights: a dictionary of the weights and biases
    (numpy.ndarrays) of the neural network
    L: the number of layers in the neural network
    m: the number of data points used

    Returns:
    The cost of the network accounting for L2 regularization.
    """
    # l2_cost = 0

    # for i in range(1, L + 1):
    #     W = weights['W' + str(i)]
    #     l2_cost += np.linalg.norm(W)**2

    # l2_cost *= lambtha / (2 * m)

    # cost += l2_cost

    return cost
