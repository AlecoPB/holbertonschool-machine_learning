#!/usr/bin/env python3
"""
Bayesian Information Criterion w/ GMMs
"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def calculate_bic(data, min_clusters=1, max_clusters=None, max_iterations=1000, convergence_tolerance=1e-5, display=False):
    """
    Determines the optimal number of clusters for a Gaussian Mixture Model using the
    Bayesian Information Criterion (BIC).
    """
    if (
        not isinstance(data, np.ndarray) or data.ndim != 2
        or not isinstance(min_clusters, int) or min_clusters <= 0
        or max_clusters is not None and (not isinstance(max_clusters, int) or max_clusters < min_clusters)
        or not isinstance(max_iterations, int) or max_iterations <= 0
        or isinstance(max_clusters, int) and max_clusters <= min_clusters
        or not isinstance(convergence_tolerance, float) or convergence_tolerance < 0
        or not isinstance(display, bool)
    ):
        return None, None, None, None

    num_samples, num_dimensions = data.shape
    if max_clusters is None:
        # If max_clusters is not defined, set it to the maximum possible
        max_clusters = num_samples
    if not isinstance(max_clusters, int) or max_clusters < 1 or max_clusters < min_clusters or max_clusters > num_samples:
        return None, None, None, None

    bic_values = []
    log_likelihoods = []

    # Iterate over each cluster size from min_clusters to max_clusters
    for current_clusters in range(min_clusters, max_clusters + 1):
        # Fit the GMM with the current cluster size
        priors, means, covariances, g, log_likelihood = expectation_maximization(
            data, current_clusters, max_iterations, convergence_tolerance, display)

        if priors is None or means is None or covariances is None or g is None:
            return None, None, None, None
        
        # Calculate the number of parameters: k * d for means,
        # k * d * (d + 1) for covariance matrices, and k - 1 for priors
        num_parameters = (current_clusters * num_dimensions) + (current_clusters * num_dimensions * (num_dimensions + 1) // 2) + (current_clusters - 1)
        bic = num_parameters * np.log(num_samples) - 2 * log_likelihood

        # Store log likelihood and BIC value for the current cluster size
        log_likelihoods.append(log_likelihood)
        bic_values.append(bic)

        # Check if the current BIC is the best observed
        if current_clusters == min_clusters or bic < best_bic:
            # Update the best values
            best_bic = bic
            optimal_result = (priors, means, covariances)
            optimal_k = current_clusters

    log_likelihoods = np.array(log_likelihoods)
    bic_values = np.array(bic_values)
    return optimal_k, optimal_result, log_likelihoods, bic_values