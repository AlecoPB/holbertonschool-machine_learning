#!/usr/bin/env python3
"""
This is some documentation
"""
import numpy as np


def precision(confusion):
    """
    Precision function for a confusion matrix
    """
    preci = np.zeros(confusion.shape[0])
    for i in range(confusion.shape[0]):
        correct = confusion[i][i]
        total = sum(confusion[:, i])
        preci[i] = round(correct / total, 8)
    return preci
