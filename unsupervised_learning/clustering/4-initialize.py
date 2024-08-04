#!/usr/bin/env python3
"""
This is some documentation
"""
import numpy as np


def initialize(X, k):
    """
    Initializes variables on a Gaussian Mixture Model.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None

    n, d = X.shape

    kmeans = __import__('1-kmeans').kmeans

    # Initialize pi evenly
    pi = np.full(shape=(k,), fill_value=1/k)

    # Initialize m using K-means
    m, _ = kmeans(X, k)

    # Initialize S as identity matrices
    S = np.tile(np.identity(d), (k, 1, 1))

    return pi, m, S
