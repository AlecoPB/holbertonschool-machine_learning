#!/usr/bin/env python3
"""
This is some documentation
"""
import numpy as np


def maximization(X, g):
    """
    Calculates the maximization step in the EM algorithm on a GMM.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None
    n, d = X.shape
    k, n_g = g.shape
    if n != n_g:
        return None, None, None
    if not np.allclose(np.sum(g, axis=0), np.ones(n)):
        return None, None, None

    # Calculate the updated priors
    pi = np.sum(g, axis=1) / n

    # Calculate the updated means
    m = np.dot(g, X) / np.sum(g, axis=1)[:, np.newaxis]

    # Calculate the updated covariance matrices
    S = np.zeros((k, d, d))
    for i in range(k):
        X_centered = X - m[i]
        S[i] = np.dot(g[i] * X_centered.T, X_centered) / np.sum(g[i])

    return pi, m, S
