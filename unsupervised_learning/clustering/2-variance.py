#!/usr/bin/env python3
"""
This is some documentation
"""
import numpy as np


def variance(X, C):
    """
    Returns:
    float: The total variance, or None on failure.
    """
    try:
        # Calculate the distances from each point to each centroid
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)

        min_distances = np.min(distances, axis=1)

        # Calculate the total variance
        var = np.sum(min_distances ** 2)

        return var
    except Exception:
        return None
