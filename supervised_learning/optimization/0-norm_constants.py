#!/usr/bin/env python3
"""

Returns:
    _type_: _description_
"""
import numpy as np


def normalization_constants(X):
    """
    Calculate the normalization constants of a matrix.

    Parameters:
    X (numpy.ndarray): The input matrix of shape (m, nx) to normalize.
                       m is the number of data points.
                       nx is the number of features.

    Returns:
    mean (numpy.ndarray): The mean of each feature.
    std_dev (numpy.ndarray): The standard deviation of each feature.
    """
    # Calculate the mean of each feature
    mean = np.mean(X, axis=0)

    # Calculate the standard deviation of each feature
    std_dev = np.std(X, axis=0)

    return mean, std_dev
