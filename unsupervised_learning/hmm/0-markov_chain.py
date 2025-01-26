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
    if not isinstance(P, np.ndarray) or not isinstance(s, np.ndarray):
        return None
    if P.ndim != s.ndim:
        return None
    if s.shape[1] != P.shape[0] or s.shape[1] != P.shape[1]:
        return None
    if not isinstance(t, int) or t < 0:
        return None
    if not np.allclose(P.sum(axis=1), 1) or not np.isclose(s.sum(), 1):
        return None

    # k-th state matrix: S{k} = S{0}.P^k, where k is equal to t in this case
    return np.dot(s, np.linalg.matrix_power(P, t))
