#!/usr/bin/env python3
"""
This is some documentation
"""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5,
                             verbose=False):
    """
    Performs the expectation maximization for a GMM
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    g, l = expectation(X, pi, m, S)

    for i in range(iterations):
        g, l_new = expectation(X, pi, m, S)
        if np.abs(l_new - l) <= tol:
            break
        pi, m, S = maximization(X, g)
        l = l_new
        if verbose and (i % 10 == 0 or i == iterations - 1):
            print(f"Log Likelihood after {i} iterations: {l:.5f}")

    g, l = expectation(X, pi, m, S)
    if verbose:
        print(f"Log Likelihood after {i + 1} iterations: {l:.5f}")

    return pi, m, S, g, l