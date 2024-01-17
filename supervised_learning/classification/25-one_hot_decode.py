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
    print(one_hot)
    cond = not isinstance(one_hot, np.ndarray) or\
        one_hot.ndim < 2 or np.any(one_hot.sum(axis=0) != 1)

    if cond:
        return None

    try:
        labels = np.argmax(one_hot, axis=0)
        return labels

    except Exception as e:
        return None
