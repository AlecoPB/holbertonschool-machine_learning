#!/usr/bin/env python3
"""
Agglomerative Clustering
"""

import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def perform_agglomerative_clustering(data, max_distance):
    """
    Executes agglomerative clustering on a dataset and visualizes the resulting
    dendrogram, with each cluster represented in a distinct color.
    """
    # Conduct hierarchical/agglomerative clustering using Ward's method
    linkage_matrix = scipy.cluster.hierarchy.linkage(data, method="ward")

    plt.show()

    # Return cluster assignments based on the specified distance threshold
    return scipy.cluster.hierarchy.fcluster(Z=linkage_matrix,
                                            t=max_distance, criterion="distance")
