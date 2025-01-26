#!/usr/bin/env python3
"""
This is some documentation
"""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Executes the expectation-maximization (EM) algorithm for a Gaussian Mixture
    Model (GMM) on a specified dataset.
    """
    if (
        not isinstance(X, np.ndarray) or X.ndim != 2
        or not isinstance(k, int) or k <= 0
        or not isinstance(iterations, int) or iterations <= 0
        or not isinstance(tol, float) or tol < 0
        or not isinstance(verbose, bool)
    ):
        return None, None, None, None, None

    # Initialize priors, means, and covariance matrices
    pi, m, S = initialize(X, k)

    for i in range(iterations):
        # Calculate probabilities and likelihoods with current parameters
        g, prev_li = expectation(X, pi, m, S)

        # Print the likelihood every 10 iterations if verbose is enabled
        if verbose and i % 10 == 0:
            print(f"Log Likelihood after {i} iterations: {round(prev_li, 5)}")

        # Update the parameters with the new estimates
        pi, m, S = maximization(X, g)

        # Calculate the new log likelihood
        g, li = expectation(X, pi, m, S)

        # Stop if the change in likelihood is below the tolerance
        if np.abs(li - prev_li) <= tol:
            break

    # Final verbose message with the current likelihood
    if verbose:
        # NOTE: i + 1 since it has been updated once more since the last print
        print(f"Log Likelihood after {i + 1} iterations: {round(li, 5)}")
    return pi, m, S, g, li
