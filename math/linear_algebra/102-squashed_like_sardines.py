#!/usr/bin/env python3
"""
This is some documentation
"""


def cat_matrices(mat1, mat2, axis=0):
    # Check if the matrices can be concatenated along the given axis
    if len(mat1.shape) != len(mat2.shape) or \
       any(s1 != s2 for s1, s2 in zip(mat1.shape, mat2.shape) if s1 != axis and s2 != axis):
        return None

    # Create a new matrix to hold the result
    shape = list(mat1.shape)
    shape[axis] += mat2.shape[axis]
    result = [[0] * shape[1] for _ in range(shape[0])]

    # Copy the elements from the first matrix
    for i in range(mat1.shape[0]):
        for j in range(mat1.shape[1]):
            result[i][j] = mat1[i][j]

    # Copy the elements from the second matrix
    for i in range(mat2.shape[0]):
        for j in range(mat2.shape[1]):
            if axis == 0:
                result[i + mat1.shape[0]][j] = mat2[i][j]
            else:
                result[i][j + mat1.shape[1]] = mat2[i][j]

    return result