#!/usr/bin/env python3
"""
This is some documentation
"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Epsilon greedy implementation
    """
    # Random number to decide if explore or exploit
    p = np.random.uniform(0, 1)

    # Check if p is less than our epsilon hyperparamenter
    if p < epsilon:
        action = np.random.randint(0, Q.shape[1])  # Explore
    else:
        action = np.argmax(Q[state])  # Eploit

    return action
