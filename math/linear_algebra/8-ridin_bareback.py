#!/usr/bin/env python3
"""
This is some documentation
"""


def mat_mul(mat1, mat2):
    """_summary_

    Args:
        mat1 (_type_): _description_
        mat2 (_type_): _description_

    Returns:
        _type_: _description_
    """
    if len(mat1[0]) != len(mat2):
        return None

    new_matrix = []
    for i in range(len(mat1)):
        new_matrix.append([])
        for j in range(len(mat2[0])):
            sum = 0
            for n in range(len(mat1[0])):
                sum += mat1[i][n] * mat2[n][j]
            new_matrix[i].append(sum)
    return new_matrix
