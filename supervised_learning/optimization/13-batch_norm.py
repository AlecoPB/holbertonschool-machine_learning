#!/usr/bin/env python3
"""_summary_
This is some documentation
"""
import numpy as np


import numpy as np

def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalizes an unactivated output of a neural network using batch normalization.

    Parameters:
    Z (numpy.ndarray): The input matrix of shape (m, n) that should be normalized.
    gamma (numpy.ndarray): The scale parameters for batch normalization, of shape (1, n).
    beta (numpy.ndarray): The offset parameters for batch normalization, of shape (1, n).
    epsilon (float): A small number used to avoid division by zero.

    Returns:
    Z_norm (numpy.ndarray): The normalized Z matrix.
    """
    # Calculate the mean and variance of each feature
    mu = np.mean(Z, axis=0)
    var = np.var(Z, axis=0)

    # Normalize the features
    Z_centered = Z - mu
    Z_norm = Z_centered / np.sqrt(var + epsilon)

    # Scale and shift the normalized features
    out = gamma * Z_norm + beta

    return out
