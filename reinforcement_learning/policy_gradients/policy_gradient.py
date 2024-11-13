#!/usr/bin/env python3
"""
This is some documenation
"""
import numpy as np


def policy(matrix, weight):
    """
    Compute a policy action based on a state matrix and a weight matrix.
    
    Parameters:
    - matrix: np.ndarray, the input matrix of state vectors (each row as a state).
    - weight: np.ndarray, the matrix of weights to apply to each state vector.
    
    Returns:
    - np.ndarray: the computed actions or policy probabilities for each state.
    """
    # Compute the dot product of each state in 'matrix' with 'weight'
    policy_output = np.dot(matrix, weight)
    
    # Apply softmax to each row to interpret results as probabilities
    exp_values = np.exp(policy_output)  # For numerical stability
    policy_probabilities = exp_values / np.sum(exp_values)
    
    return policy_probabilities

def policy_gradient(state, weight):
    """
    Compute the Monte-Carlo policy gradient based on a state and a weight matrix.
    
    Parameters:
    - state: np.ndarray, the matrix representing the current observation of the environment.
    - weight: np.ndarray, the matrix of random weights.
    
    Returns:
    - tuple: (action, gradient) where action is the selected action and gradient is the policy gradient.
    """
    n_states, n_actions = weight.shape

    gradient = np.zeros((n_states, n_actions))
    policy_value = policy(state, weight)

    if np.random.random() > policy_value[0]:
        action = 1
    else:
        action = 0

    # Compute the gradient
    for n in range(n_actions):
        if n == action:
            gradient[:, n] = state * (1 - policy_value[n])
        else:
            gradient[:, n] = -state * policy_value[n]
        
    return action, gradient
