#!/usr/bin/env python3
"""
This is some documentation
"""
import numpy as np

def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None
    if kmax is not None and (not isinstance(kmax, int) or kmax <= 0):
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None
    if kmax is None:
        kmax = X.shape[0]
    if kmax < kmin:
        return None, None

    kmeans = __import__('1-kmeans').kmeans
    variance = __import__('2-variance').variance

    results = []
    d_vars = []

    for k in range(kmin, kmax + 1):
        centroids, clss = kmeans(X, k, iterations)
        results.append((centroids, clss))
        var = variance(X, centroids)
        if k == kmin:
            min_var = var
        d_vars.append(min_var - var)

    return results, d_vars
