#!/usr/bin/env python3
"""
This is some documentation
"""


def np_slice(matrix, axes={}):
    """_summary_

    Args:
        matrix (_type_): _description_
        axes (dict, optional): _description_. Defaults to {}.

    Returns:
        _type_: _description_
    """
    new_arr = []
    for key, value in axes.items():
        val1, val2 = value[0], value[1]
        new_arr.append(matrix[key][val1:val2])
    return new_arr
