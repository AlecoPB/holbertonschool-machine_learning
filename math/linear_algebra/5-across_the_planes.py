#!/usr/bin/env python3
matrix_shape = __import__('2-size_me_please').matrix_shape
add_arrays = __import__('4-line_up').add_arrays
"""
This module only has one method
"""


def add_matrices2D(mat1, mat2):
    """
    Returns the sum of two matrices.
    """
    if matrix_shape(mat1) != matrix_shape(mat2):
        return None

    mat_Sum = []
    for i in range(len(mat1)):
        mat_Sum.append(add_arrays(mat1[i], mat2[i]))
    return mat_Sum

