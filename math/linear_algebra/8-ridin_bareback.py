#!/usr/bin/env python3
"""
This module has only one method
"""


def mat_mul(mat1, mat2):
    """
    This method multiplies two matrices
    """
    if len(mat1[0]) != len(mat2):
        return None

    mat_mul = []
    line = 0
    for L in range(len(mat1)):
        mat_mul.append([])
        for C in range(len(mat2[0])):
            line = 0
            for i in range(len(mat2)):
                line += mat1[L][i] * mat2[i][C]
            mat_mul[L].append(line)

    return mat_mul
