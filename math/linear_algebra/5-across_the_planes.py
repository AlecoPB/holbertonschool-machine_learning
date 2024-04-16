#!/usr/bin/env python3
"""
This is some documentation
"""


def add_matrices2D(mat1, mat2):
    """_summary_

    Args:
        mat1 (_type_): _description_
        mat2 (_type_): _description_

    Returns:
        _type_: _description_
    """
    if len(mat1) != len(mat2) or\
        len(mat1[0]) != len(mat2[0]):
            None
    else:
        new_mat = []
        for i in range(len(mat1)):
            new_mat.append([])
            for j in range(len(mat1[0])):
                new_mat[i].append(mat1[i][j]
                                  + mat2[i][j])
        return new_mat
