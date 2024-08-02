#!/usr/bin/env python3
"""
This is some documentation
"""
import numpy as np


def kmeans(X, k, iterations=1000):
    """
    performs K-means on a dataset
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if type(k) is not int or k <= 0:
        return None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None
    n, d = X.shape
    low = X.min(axis=0)
    high = X.max(axis=0)
    centroids = np.random.uniform(low, high, (k, d))
    for _ in range(iterations):
        copy = centroids.copy()
        dists = np.linalg.norm(X[:, None] - centroids, axis=-1)
        clusters = np.argmin(dists, axis=-1)
        for i in range(k):
            if len(X[clusters == i]) == 0:
                centroids[i] = np.random.uniform(low, high)
            else:
                centroids[i] = X[clusters == i].mean(axis=0)
        if np.all(copy == centroids):
            break
    dists = np.linalg.norm(X[:, None] - centroids, axis=-1)
    clusters = np.argmin(dists, axis=-1)
    return centroids, clusters
