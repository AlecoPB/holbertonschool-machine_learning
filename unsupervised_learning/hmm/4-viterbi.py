#!/usr/bin/env python3
"""
This is some documentation
"""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """
    Most likely sequence of hidden states for a hidden markov model
    """
    try:
        T = Observation.shape[0]  # Number of observations
        N = Emission.shape[0]      # Number of hidden states

        # Step 1: Initialize V and backpointer matrices
        V = np.zeros((N, T))
        backpointer = np.zeros((N, T), dtype=int)

        # Initial step
        V[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]

        # Step 2: Recursion
        for t in range(1, T):
            for j in range(N):
                probabilities = V[:, t-1] * Transition[:, j]
                V[j, t] = np.max(probabilities) * Emission[j, Observation[t]]
                backpointer[j, t] = np.argmax(probabilities)

        # Step 3: Termination
        P = np.max(V[:, T-1])
        best_last_state = np.argmax(V[:, T-1])

        # Step 4: Path reconstruction
        path = [0] * T
        path[-1] = best_last_state
        for t in range(T-2, -1, -1):
            path[t] = backpointer[path[t+1], t+1]

        return path, P

    except Exception:
        return None, None
