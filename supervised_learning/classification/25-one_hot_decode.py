#!/usr/bin/env python3
"""
_summary_
Method to decode a one_hot matrix
"""
import numpy as np


def one_hot_decode(one_hot):
    """_summary_

    Args:
        one_hot (np.ndarray): One_hot encoded matrix
        with shape (classes, m)
    """
    if not isinstance(one_hot, np.ndarray) or len(one_hot) == 0:
        return None
    try:
        return np.argmax(one_hot, axis=0)
    except Exception as e:
        return None
