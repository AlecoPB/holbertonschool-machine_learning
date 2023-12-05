#!/usr/bin/env python3
"""
This module only has one method
"""


def matrix_transpose(matrix):
    """
    Returns the transposed of a matrix.
    """

    if not matrix:
        return []
    t_mat = []
    for row in range(len(matrix[0])):
        t_mat.append([])
        for col in range(len(matrix)):
            t_mat[row].append(matrix[col][row])
    return t_mat
