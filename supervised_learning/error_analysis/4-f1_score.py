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

    sens = sensitivity(confusion)
    prec = precision(confusion)
    
    f1_scores = np.zeros(confusion.shape[0])
    for i in range(confusion.shape[0]):
        f1_scores[i] = (2 * (prec[i] * sens[i])) / (prec[i] + sens[i])

    return f1_scores
