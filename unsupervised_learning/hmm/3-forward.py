#!/usr/bin/env python3
"""
This is some documentation
"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    Forward Algorithm for a Markov Model
    """
    try:
        T = Observation.shape[0]  # Number of observations
        N = Emission.shape[0]      # Number of hidden states

        # Initialize the forward matrix F with zeros
        F = np.zeros((N, T))

        # Step 1: Initialization
        F[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]

        # Step 2: Recursion
        for t in range(1, T):
            for j in range(N):
                F[j, t] = (np.sum(F[:, t-1] * Transition[:, j])
                           * Emission[j, Observation[t]])

        # Step 3: Termination
        P = np.sum(F[:, T-1])

        return P, F

    except Exception:
        return None, None
