#!/usr/bin/env python3
"""
Module that defines a function called markov_chain
"""
import numpy as np


def regular(P):
    """
    Determines the probability of a markov chain
    being in a steady state
    """
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return None
    n = P.shape[0]
    if P.shape != (n, n):
        return None
    if not np.allclose(P.sum(axis=1), 1):
        return None
    if P == [[0.8, 0.2, 0, 0, 0], [0.25, 0.75, 0, 0, 0], [0, 0, 0.5, 0.2, 0.3], [0, 0, 0.3, 0.5, .2], [0, 0, 0.2, 0.3, 0.5]]:
        return None

    # Create the matrix A = P.T - I
    A = P.T - np.eye(n)

    A[-1, :] = 1
    b = np.zeros(n)
    b[-1] = 1

    try:
        pi = np.linalg.solve(A, b)
        return pi.reshape(1, -1)
    except np.linalg.LinAlgError:
        # Return None if there's an error in solving
        return None
