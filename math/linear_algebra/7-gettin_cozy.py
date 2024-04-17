#!/usr/bin/env python3
"""
This is some documentation
"""


def cat_matrices2D(mat1, mat2, axis=0):
    if axis == 1 and len(mat1) != len(mat2):
        return None
    elif axis == 0 and len(mat1[0]) != len(mat2[0]):
        return None
    else:
        new_matrix = []
        if axis == 1:
            for i in range(len(mat2)):
                new_matrix.append(mat1[i] + mat2[i])
            return new_matrix
        else:
            for i in range(len(mat1) + len(mat2)):
                if i < len(mat1):
                    new_matrix.append(mat1[i])
                else:
                    new_matrix.append(mat2[i-len(mat1)])
            return new_matrix
