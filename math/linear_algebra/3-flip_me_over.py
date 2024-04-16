#!/usr/bin/env python3
"""
This is some documentation
"""


def matrix_transpose(matrix):
    """
    This fucntion returns the tranpose
    of a 2D matrix
    """
    length = len(matrix[0])
    width = len(matrix)
    transpose = []
    for i in range(0, length):
        transpose.append([])
        for j in range(0, width):
            transpose[i].append(matrix[j][i])

    return transpose
