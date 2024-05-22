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
    sensi = []
    np.ndarray(sensi)
    for i in range(confusion.shape[0]):
        correct = confusion[i][i]
        incorrect = sum(confusion[i, :]) - correct
        sensi.append(round(np.float32(correct / (correct + incorrect)), 8))

    return sensi
