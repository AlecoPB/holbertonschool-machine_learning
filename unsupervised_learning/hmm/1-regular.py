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
    try:
        pi = [1]
        pi = (pi.append(0) for _ in range(P[0] - 1))
        result = np.linalg.solve(P, pi)
        return result
    except:
        return None
