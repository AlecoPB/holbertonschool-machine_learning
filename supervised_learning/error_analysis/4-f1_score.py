#!/usr/bin/env python3
"""
This is some documentation
"""
import numpy as np


def f1_score(confusion):
    """
    This function computes the f1_score of a confusion matrix
    """
    sensitivity = __import__('1-sensitivity').sensitivity
    precision = __import__('2-precision').precision

    e = 1e-7
    print(e)
    sens = sensitivity(confusion)
    prec = precision(confusion)

    f1_scores = 2 * ((prec * sens) / (prec + sens + e))

    return f1_scores
