#!/usr/bin/env python3
"""
This is some documentation
"""
import numpy as np


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Tests the optimum number of clusters by variance.
    """
    if kmax is None:
        kmax = X.shape[0]

    if not isinstance(kmin, int) or not isinstance(kmax, int) or not isinstance(iterations, int):
        return None, None
    if kmin <= 0 or kmax <= 0 or iterations <= 0:
        return None, None
    if kmin >= kmax:
        return None, None

    kmeans = __import__('1-kmeans').kmeans
    variance = __import__('2-variance').variance

    results = []
    d_vars = []

    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        results.append((C, clss))
        var = variance(X, C)
        d_vars.append(var)

    min_var = d_vars[0]
    d_vars = [var - min_var for var in d_vars]

    return results, d_vars
