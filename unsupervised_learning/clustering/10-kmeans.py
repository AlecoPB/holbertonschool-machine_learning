#!/usr/bin/env python3
"""
This is some documentation
"""
import sklearn.cluster


def kmeans(X, k):
    """
    Performs K-means on a dataset.
    """
    if not hasattr(X, "shape") or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None

    kmeans = sklearn.cluster.KMeans(n_clusters=k)
    kmeans.fit(X)
    C = kmeans.cluster_centers_
    clss = kmeans.labels_

    return C, clss
