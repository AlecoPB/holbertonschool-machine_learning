#!/usr/bin/env python3
"""
This is some documenation
"""
import numpy as np


def policy(matrix, weight):
    """
    Compute a policy action based on a state matrix and a weight matrix.

    Parameters:
    - matrix: np.ndarray, the input matrix of
    state vectors (each row as a state).
    - weight: np.ndarray, the matrix of weights
    to apply to each state vector.

    Returns:
    - np.ndarray: the computed actions or policy
    probabilities for each state.
    """
    # Compute the dot product of each state in
    # 'matrix' with 'weight'
    policy_output = np.dot(matrix, weight)

    # Apply softmax to each row to interpret
    # results as probabilities
    exp_values = np.exp(policy_output -
                        np.max(policy_output,
                               axis=1,
                               keepdims=True))
    policy_probabilities = exp_values / np.sum(exp_values,
                                               axis=1,
                                               keepdims=True)

    return policy_probabilities
