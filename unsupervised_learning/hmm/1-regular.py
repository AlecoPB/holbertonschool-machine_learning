#!/usr/bin/env python3
"""
Module that defines a function called markov_chain
"""
import numpy as np


def regular(P, s, t=1):
    """
    Determines the probability of a markov chain
    being in a steady state
    """
    try:
        result = s
        steady = []
        while result != steady:
            steady = result
            result = np.matmul(result, P)
        return result
    except:
        return None
