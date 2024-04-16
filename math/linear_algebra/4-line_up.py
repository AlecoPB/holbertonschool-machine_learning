#!/usr/bin/env python3
"""
This is some documentation
"""


def add_arrays(arr1, arr2):
    """_summary_

    Args:
        arr1 (_type_): _description_
        arr2 (_type_): _description_

    Returns:
        _type_: _description_
    """
    new_arr = []
    if len(arr1) != len(arr2):
        return None
    else:
        for i in range(len(arr1)):
            new_arr.append(arr1[i] + arr2[i])
        return new_arr
