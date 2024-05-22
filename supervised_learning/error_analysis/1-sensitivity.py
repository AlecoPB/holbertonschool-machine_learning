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
    correct, incorrect = 0, 0
    sensi = []
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[0]):
            if i==j:
                correct += confusion[i][j]
            else:
                incorrect += confusion[j][i]
            sensi.append(correct / (correct + incorrect))
    return sensi
