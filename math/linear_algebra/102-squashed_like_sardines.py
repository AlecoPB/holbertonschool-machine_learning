#!/usr/bin/env python3
"""
This is some documentation
"""


def cat_matrices(mat1, mat2, axis=0):
    """
    Concatenates matrices
    """
    if len(mat1) != len(mat2) and axis == 0:
        return None
    if isinstance(mat1[0], list) and isinstance(mat2[0], list) and len(mat1[0]) != len(mat2[0]) and axis == 1:
        return None

    if axis == 0:
        return mat1 + mat2
    else:
        return [row1 + row2 for row1, row2 in zip(mat1, mat2)]
