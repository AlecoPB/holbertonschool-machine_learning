#!/usr/bin/env python3
"""
This module has only one method
"""


def np_elementwise(mat1, mat2):
    """
    This method does basic operations
    """
    add = mat1 + mat2
    sub = mat1 - mat2
    mul = mat1 * mat2
    div = mat1 / mat2
    return add, sub, mul, div
