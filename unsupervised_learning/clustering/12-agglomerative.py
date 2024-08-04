#!/usr/bin/env python3
"""
This is some documentation
"""
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    Performs agglomerative clustering on a dataset.
    """
    # Perform hierarchical/agglomerative clustering using Ward linkage
    Z = sch.linkage(X, method='ward')

    # Create the dendrogram
    plt.figure()
    sch.dendrogram(Z, color_threshold=dist)
    plt.show()

    # Form flat clusters
    clss = sch.fcluster(Z, t=dist, criterion='distance') - 1

    return clss
