#!/usr/bin/env python3
"""
This is some documentation
"""
import numpy as np


def sensitivity(confusion):
    """_summary_

    Args:
        confusion (matrix): Confusion matrix 

    Returns:
        int: sensitivity value
    """
    sensi = np.zeros(confusion.shape[0])
    for i in range(confusion.shape[0]):
        correct = confusion[i][i]
        incorrect = sum(confusion[i, :]) - correct
        sensi[i] = round(correct / (correct + incorrect), 8)

    return sensi
