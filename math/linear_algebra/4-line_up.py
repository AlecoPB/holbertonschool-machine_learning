#!/usr/bin/env python3
"""
This module only has one method
"""


def add_arrays(arr1, arr2):
    """
    Returns the sum of two arrays.
    """
    if len(arr1) != len(arr2):
        return None
    for i in range(len(arr1)):
        if not isinstance(arr1[i], int) or not isinstance(arr2[i], int):
            raise TypeError("add_arrays() only accepts arrays of integers.")
  
    arr_Sum = []
    for i in range(len(arr1)):
        arr_Sum.append(arr1[i] + arr2[i])
    return arr_Sum
