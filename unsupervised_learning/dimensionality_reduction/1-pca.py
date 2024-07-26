#!/usr/bin/env python3
"""
This module contains a function to perform PCA on a dataset.
"""
import numpy as np


def pca(X, ndim):
    """
    Performs PCA on a dataset.
    """
    # Center the data by subtracting the mean of each feature
    X_centered = X - np.mean(X, axis=0)

    # Compute the covariance matrix
    cov_matrix = np.cov(X_centered, rowvar=False)

    # Perform eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort the eigenvalues and corresponding eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Select the top ndim eigenvectors
    W = sorted_eigenvectors[:, :ndim]

    # Transform the data
    T = np.dot(X_centered, W)

    return T
