#!/usr/bin/env python3
"""
_summary_
Method to decode a one_hot matrix
"""
import numpy as np


def one_hot_decode(one_hot):
    """_summary_

    Args:
        one_hot (_type_): _description_

    Returns:
        _type_: _description_
    """
    if not isinstance(one_hot, np.ndarray) or one_hot.ndim < 2:
        return None
    elif np.any(one_hot < 0):
        return None
    elif np.any(one_hot.sum(axis=1) > 1):
        return None

    try:
        labels = np.argmax(one_hot, axis=0)
        return labels
    
    except Exception as e:
        return None
