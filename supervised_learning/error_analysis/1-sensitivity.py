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
    for i in range(confusion.shape[0]):
        correct = confusion[i][i]
        incorrect = sum(confusion[i, :]) - correct
        sensi.append(np.float16(correct / (correct + incorrect)))

    return sensi
