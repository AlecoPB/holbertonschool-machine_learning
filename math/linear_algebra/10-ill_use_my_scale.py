#!/usr/bin/env python3
"""
This is some documentation
"""


def np_shape(matrix):
    """_summary_

    Args:
        matrix (_type_): _description_

    Returns:
        _type_: _description_
    """
    if len(matrix) == 0:
        return (0,)
    elif not isinstance(matrix[0], np.ndarray):
        return (len(matrix),)
    else:
        return (len(matrix),) + np_shape(matrix[0])
