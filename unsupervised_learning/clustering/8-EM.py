#!/usr/bin/env python3
"""
This is some documentation
"""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization

def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
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

    try:
        pi, m, S = initialize(X, k)
        g, l_prev = expectation(X, pi, m, S)

        for i in range(iterations):
            g, l = expectation(X, pi, m, S)
            pi, m, S = maximization(X, g)

            if verbose and i % 10 == 0:
                print(f"Log Likelihood after {i} iterations: {l:.5f}")

            if abs(l - l_prev) <= tol:
                break

            l_prev = l

        g, l = expectation(X, pi, m, S)
        if verbose:
            print(f"Log Likelihood after {i + 1} iterations: {l:.5f}")

        return pi, m, S, g, l
    except Exception:
        return None, None, None, None, None
