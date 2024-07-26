#!/usr/bin/env python3
"""
This module contains a function to perform PCA on a dataset.
"""
import numpy as np


np.set_printoptions(precision=4, suppress=False, formatter={'float': '{:0.4e}'.format})
def pca(X, var=0.95):
	"""
	Performs PCA on a dataset.
	"""
	# Compute the covariance matrix
	cov_matrix = np.dot(X.T, X) / X.shape[0]

	# Perform eigen decomposition
	eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

	# Sort the eigenvalues and corresponding eigenvectors in descending order
	sorted_indices = np.argsort(eigenvalues)[::-1]
	sorted_eigenvalues = eigenvalues[sorted_indices]
	sorted_eigenvectors = eigenvectors[:, sorted_indices]

	# Compute the cumulative variance
	cumulative_variance = np.cumsum(sorted_eigenvalues) / np.sum(sorted_eigenvalues)

	# Determine the number of components to maintain the desired variance
	nd = np.searchsorted(cumulative_variance, var) + 1

	# Select the top nd eigenvectors
	W = sorted_eigenvectors[:, :nd]

	return W
