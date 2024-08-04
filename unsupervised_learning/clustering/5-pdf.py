#!/usr/bin/env python3
"""
This is some documentation
"""
import numpy as np


def pdf(X, m, S):
    """
    Calculates the probability density function of a Gaussian distribution.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None
    if not isinstance(S, np.ndarray) or len(S.shape) != 2:
        return None
    if X.shape[1] != m.shape[0] or S.shape[0] != S.shape[1] or S.shape[0] != m.shape[0]:
        return None

    n, d = X.shape

    # Calculate the determinant and inverse of the covariance matrix
    det_S = np.linalg.det(S)
    inv_S = np.linalg.inv(S)

    # Calculate the normalization factor
    norm_factor = 1 / np.sqrt((2 * np.pi) ** d * det_S)

    # Calculate the exponent term
    diff = X - m
    exponent = -0.5 * np.sum(diff @ inv_S * diff, axis=1)

    # Calculate the PDF values
    P = norm_factor * np.exp(exponent)

    # Ensure all values in P have a minimum value of 1e-300
    P = np.maximum(P, 1e-300)

    return P
