#!/usr/bin/env python3
"""
Method to convert a numeric label vector into a one-hot matrix
"""
import numpy as np


def one_hot_encode(Y, classes):
    """
    Convert a numeric label vector into a one-hot matrix

    Args:
    Y (numpy.ndarray): a numpy array with shape
    (m,) containing numeric class labels
    classes (int): the maximum number of classes found in Y

    Returns:
    numpy.ndarray: a one-hot encoding of Y with shape
    (classes, m), or None on failure
    """
    if not isinstance(Y, np.ndarray) or len(Y) == 0:
        return None
    if not isinstance(classes, int) or classes <= np.max(Y):
        return None
    try:
        one_hot = np.zeros((classes, Y.shape[0]))
        one_hot[Y, np.arange(Y.shape[0])] = 1
        return one_hot
    except Exception as e:
        return None
