#!/usr/bin/env python3
"""
Module that defines a function called markov_chain
"""
import numpy as np


def markov_chain(P, s, t=1):
    """
    Determines the probability of a markov chain
    being in a particular state
    """
    result = s
    for _ in range(t):
        result = np.matmul(result, P)
    return result
