#!/usr/bin/env python3
"""
Bayesian Information Criterion w/ GMMs
"""


import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Finds the best number of clusters for a Gaussian Mixture Model using the
    Bayesian Information Criterion (BIC).
    """
    if (
        not isinstance(X, np.ndarray) or X.ndim != 2
        or not isinstance(kmin, int) or kmin <= 0
        or kmax is not None and (not isinstance(kmax, int) or kmax < kmin)
        or not isinstance(iterations, int) or iterations <= 0
        or isinstance(kmax, int) and kmax <= kmin
        or not isinstance(iterations, int) or iterations <= 0
        or not isinstance(tol, float) or tol < 0
        or not isinstance(verbose, bool)
    ):
        return None, None, None, None

    n, d = X.shape
    if kmax is None:
        kmax = n
    if not isinstance(kmax, int) or kmax < 1 or kmax < kmin or kmax > n:
        return None, None, None, None

    likelihoods = []
    b = []
    best_bic, best_results, best_k = None, None, None

    for k in range(kmin, kmax + 1):
        result = expectation_maximization(X, k, iterations, tol, verbose)
        if result is None or any(part is None for part in result):
            return None, None, None, None

        pi, m, S, g, li = result
        p = (k * d) + (k * d * (d + 1) // 2) + (k - 1)
        bic = p * np.log(n) - 2 * li

        likelihoods.append(li)
        b.append(bic)

        if best_bic is None or bic < best_bic:
            best_bic = bic
            best_results = (pi, m, S)
            best_k = k

    return best_k, best_results, np.array(likelihoods), np.array(b)
