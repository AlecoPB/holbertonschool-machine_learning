#!/usr/bin/env python3
"""
This is some documnetation
"""
import numpy as np


def initialize(X, k):
    """
    initializes cluster centroids for k-means
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if type(k) is not int or k <= 0:
        return None
    n, d = X.shape
    low = X.min(axis=0)
    high = X.max(axis=0)
    return np.random.uniform(low, high, (k, d))
