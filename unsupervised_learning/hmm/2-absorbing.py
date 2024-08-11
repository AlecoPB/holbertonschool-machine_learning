#!/usr/bin/env python3
"""
This is some documentation
"""
import numpy as np

def absorbing(P):
    """
    Determines if a Markov Chain is absorbing
    """
    n = P.shape[0]

    # Step 1: Identify absorbing states
    absorbing_states = [i for i in range(n) if P[i, i] == 1]

    # If no absorbing states, return False
    if not absorbing_states:
        return False

    # Step 2: Check if every state can reach an absorbing state
    for i in range(n):
        if i not in absorbing_states:
            # Check if there is a path from state i to any absorbing state
            reachable = False
            for j in absorbing_states:
                if np.linalg.matrix_power(P, n)[i, j] > 0:
                    reachable = True
                    break
            if not reachable:
                return False

    # If all non-absorbing states can reach an absorbing state, return True
    return True
