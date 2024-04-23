#!/usr/bin/env python3
"""
This is some documentation
"""


def cat_matrices(mat1, mat2, axis=0):
    # Check if the matrices can be concatenated along the given axis
    if len(mat1[0]) != len(mat2[0]) and axis == 0:
        return None
    if len(mat1) != len(mat2) and axis == 1:
        return None

    # Concatenate the matrices
    result = []
    if axis == 0:
        result = mat1 + mat2
    elif axis == 1:
        for m1, m2 in zip(mat1, mat2):
            result.append(m1 + m2)
    return result