#!/usr/bin/env python3 
"""
This is some documentation
"""
import numpy as np


def normalize(X, m, s):
    """
    Normalize (standardize) a matrix.

    Parameters:
    X (numpy.ndarray): The input matrix of shape (d, nx) to normalize.
                       d is the number of data points.
                       nx is the number of features.
    m (numpy.ndarray): The mean of all features of X. Shape (nx,).
    s (numpy.ndarray): The standard deviation of all features of X. Shape (nx,).

    Returns:
    X_normalized (numpy.ndarray): The normalized X matrix.
    """
    X_normalized = (X - m) / s

    return X_normalized
