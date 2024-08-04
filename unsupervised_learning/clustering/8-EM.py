#!/usr/bin/env python3
"""
This is some documentation
"""
import numpy as np

def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    try:
        initialize = __import__('4-initialize').initialize
        expectation = __import__('6-expectation').expectation
        maximization = __import__('7-maximization').maximization

        pi, m, S = initialize(X, k)
        l_prev = 0

        for i in range(iterations):
            g, l = expectation(X, pi, m, S)
            pi, m, S = maximization(X, g)

            if verbose and (i % 10 == 0 or i == iterations - 1):
                print(f"Log Likelihood after {i} iterations: {l:.5f}")

            if abs(l - l_prev) <= tol:
                break

            l_prev = l

        return pi, m, S, g, l
    except Exception:
        return None, None, None, None, None
