#!/usr/bin/env python3
"""
This is some documentation
"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Finds the best number of clusters for a GMM using the Bayesian Information Criterion.

    Parameters:
    X (np.ndarray): The data set of shape (n, d).
    kmin (int): The minimum number of clusters to check for (inclusive).
    kmax (int): The maximum number of clusters to check for (inclusive).
    iterations (int): The maximum number of iterations for the EM algorithm.
    tol (float): The tolerance for the EM algorithm.
    verbose (bool): If True, print information during the EM algorithm.

    Returns:
    tuple: (best_k, best_result, l, b) or (None, None, None, None) on failure.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None, None, None
    if kmax is not None and (not isinstance(kmax, int) or kmax <= 0):
        return None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None

    n, d = X.shape
    if kmax is None:
        kmax = n

    best_k = None
    best_result = None
    best_bic = np.inf
    log_likelihoods = []
    bics = []

    for k in range(kmin, kmax + 1):
        pi, m, S, g, l = expectation_maximization(X, k, iterations, tol, verbose)
        if pi is None or m is None or S is None or g is None or l is None:
            return None, None, None, None

        log_likelihoods.append(l)
        p = k * d + k * d * (d + 1) / 2 + k - 1
        bic = p * np.log(n) - 2 * l
        bics.append(bic)

        if bic < best_bic:
            best_bic = bic
            best_k = k
            best_result = (pi, m, S)

    return best_k, best_result, np.array(log_likelihoods), np.array(bics)