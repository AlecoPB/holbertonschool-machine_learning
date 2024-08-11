#!/usr/bin/env python3
"""
This is some documentation
"""
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """
    Backward algorithm for a Markov Model
    """
    try:
        T = Observation.shape[0]  # Number of observations
        N = Emission.shape[0]      # Number of hidden states

        # Step 1: Initialize B with zeros and set the last column to 1
        B = np.zeros((N, T))
        B[:, T-1] = 1

        # Step 2: Recursion (going backwards in time)
        for t in range(T-2, -1, -1):
            for i in range(N):
                B[i, t] = (np.sum(B[:, t+1] * Transition[i, :]
                                  * Emission[:, Observation[t+1]]))

        # Step 3: Termination
        P = np.sum(Initial[:, 0] * Emission[:, Observation[0]] * B[:, 0])

        return P, B

    except Exception:
        return None, None
