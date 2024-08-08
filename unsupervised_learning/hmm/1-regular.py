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

    # Check if rows sum to 1 (valid transition matrix)
    if not np.allclose(P.sum(axis=1), 1):
        return None

    # Subtract the identity matrix from the transpose of P
    A = P.T - np.eye(n)

    # Add the normalization condition (sum of probabilities = 1)
    A[-1, :] = 1
    b = np.zeros(n)
    b[-1] = 1

    try:
        # Solve the linear system to find steady-state probabilities
        steady_state = np.linalg.solve(A, b)
        return steady_state.reshape(1, -1)
    except np.linalg.LinAlgError:
        # If the matrix is singular or any other issue arises
        return None
