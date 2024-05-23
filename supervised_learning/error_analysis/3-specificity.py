#!/usr/bin/env python3
"""
This is some documentation
"""


import numpy as np

def specificity(confusion):
    """_summary_
    Args:
        confusion (_type_): _description_
    Returns:
        _type_: _description_
    """
    classes = confusion.shape[0]
    specificity_scores = np.zeros(classes)

    for i in range(classes):
        true_negatives = np.sum(confusion) - (np.sum(confusion[i, :]) + np.sum(confusion[:, i]) - confusion[i, i])
        false_positives = np.sum(confusion[:, i]) - confusion[i, i]
        specificity_scores[i] = true_negatives / (true_negatives + false_positives)
    
    return specificity_scores
