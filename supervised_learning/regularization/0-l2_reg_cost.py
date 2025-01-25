#!/usr/bin/env python3
"""
This is some documentation
"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the cost of a neural network with L2 regularization.

    Parameters:
    cost (float): The cost of the network without L2 regularization.
    lambtha (float): The regularization parameter.
    weights (dict): A dictionary of the weights
    and biases (numpy.ndarrays) of the neural network.
    L (int): The number of layers in the neural network.
    m (int): The number of data points used.

    Returns:
    float: The cost of the network accounting for L2 regularization.
    """
    # Initialize the L2 regularization term
    l2_reg = 0

    # Iterate through the weights dictionary
    for key in weights:
        # Only consider weights, not biases
        if 'b' not in key:  # Assuming biases are labeled with 'b'
            l2_reg += np.sum(np.square(weights[key]))

    # Calculate the L2 regularization cost
    l2_reg_cost = (lambtha / (2 * m)) * l2_reg

    # Total cost with L2 regularization
    total_cost = cost + l2_reg_cost

    return total_cost
