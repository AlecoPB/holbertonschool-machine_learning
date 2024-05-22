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
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[0]):
            if i==j:
                correct += confusion[i][j]
            else:
                incorrect += confusion[i][j]
    return correct / (correct + incorrect)
