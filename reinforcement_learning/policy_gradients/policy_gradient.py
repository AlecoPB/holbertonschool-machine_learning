#!/usr/bin/env python3
"""
This is some documenation
"""
import numpy as np


def compute_policy(state: np.ndarray, weight_matrix: np.ndarray) -> np.ndarray:
    """
    Compute a policy action based on a state and a weight matrix.
    
    Parameters:
    - state: np.ndarray, the input state vector.
    - weight_matrix: np.ndarray, the matrix of weights to apply to the state.
    
    Returns:
    - np.ndarray: the computed action or policy decision.
    """
    # Compute the policy by applying the weight matrix to the state
    policy_output = np.dot(weight_matrix, state)
    
    # Apply a softmax function to make it a probabilistic policy (if applicable)
    policy_probabilities = np.exp(policy_output) / np.sum(np.exp(policy_output))
    
    return policy_probabilities
