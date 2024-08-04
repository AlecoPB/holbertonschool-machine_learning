#!/usr/bin/env python3
"""
This is some documentation
"""
import numpy as np

def expectation_maximization(data, iterations=100, verbose=False):
    """
    Perform the Expectation-Maximization algorithm on the given data.

    Parameters:
    data (np.ndarray): The input data for the EM algorithm.
    iterations (int): The number of iterations to run the algorithm.
    verbose (bool): If True, print log likelihood at intervals.

    Returns:
    tuple: The final parameters of the model.
    """
    # Initialize parameters
    n, d = data.shape
    means = np.random.rand(2, d)
    covariances = np.array([np.eye(d)] * 2)
    weights = np.ones(2) / 2

    for i in range(iterations):
        # E-step: compute responsibilities
        responsibilities = np.zeros((n, 2))
        for k in range(2):
            responsibilities[:, k] = weights[k] * multivariate_normal.pdf(data, means[k], covariances[k])
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)

        # M-step: update parameters
        for k in range(2):
            Nk = responsibilities[:, k].sum()
            means[k] = (data * responsibilities[:, k][:, np.newaxis]).sum(axis=0) / Nk
            covariances[k] = (responsibilities[:, k][:, np.newaxis, np.newaxis] * 
                              (data - means[k]).T @ (data - means[k])) / Nk
            weights[k] = Nk / n

        # Compute log likelihood
        log_likelihood = np.sum(np.log(np.sum([weights[k] * multivariate_normal.pdf(data, means[k], covariances[k]) 
                                               for k in range(2)], axis=0)))

        # Print log likelihood if verbose
        if verbose and (i % 10 == 0 or i == iterations - 1):
            print(f"Log Likelihood after {i} iterations: {log_likelihood:.5f}")

    return means, covariances, weights
