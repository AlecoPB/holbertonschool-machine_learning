#!/usr/bin/env python3
"""_summary_
This is some documentation
"""
import numpy as np


def shuffle_data(X, Y):
    """
    Shuffle the data points in two matrices the same way.

    Parameters:
    X (numpy.ndarray): The first matrix of shape (m, nx) to shuffle.
                       m is the number of data points.
                       nx is the number of features in X.
    Y (numpy.ndarray): The second matrix of shape (m, ny) to shuffle.
                       m is the same number of data points as in X.
                       ny is the number of features in Y.

    Returns:
    X_shuffled (numpy.ndarray): The shuffled X matrix.
    Y_shuffled (numpy.ndarray): The shuffled Y matrix.
    """
    # Create a permutation
    perm = np.random.permutation(X.shape[0])

    # Shuffle X and Y in the same way
    X_shuffled = X[perm]
    Y_shuffled = Y[perm]

    return X_shuffled, Y_shuffled
