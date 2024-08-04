#!/usr/bin/env python3
"""
This is some documentation
"""
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

def agglomerative(X, dist):
    try:
        # Perform the hierarchical clustering using Ward linkage
        Z = sch.linkage(X, method='ward')
        
        # Create the dendrogram
        plt.figure()
        dendrogram = sch.dendrogram(Z, color_threshold=dist)
        plt.show()
        
        # Get the cluster indices for each data point
        clss = sch.fcluster(Z, t=dist, criterion='distance')
        
        return clss
    except Exception:
        return None
