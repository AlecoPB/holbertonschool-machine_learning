#!/usr/bin/env python3
"""
This is some documentation
"""


def cat_arrays(arr1, arr2):
    """_summary_

    Args:
        arr1 (_type_): _description_
        arr2 (_type_): _description_

    Returns:
        _type_: _description_
    """
    new_list = []
    for i in range(len(arr1)):
        new_list.append(arr1[i])
    for i in range(len(arr2)):
        new_list.append(arr2[i])
    return new_list
