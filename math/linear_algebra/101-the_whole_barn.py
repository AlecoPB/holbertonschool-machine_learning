#!/usr/bin/env python3
"""
This is some documentation
"""


def check_dim(mat1, mat2):
    """
    This function checks if the dimensions of
    both matrices are the same
    """
    if not isinstance(mat1[0], list):
        if len(mat1) == len(mat2):
            return True
        else:
            return False
    else:
        if len(mat1) == len(mat2):
            return check_dim(mat1[0], mat2[0])
        else:
            return False


def add_matrices(mat1, mat2):
    """
    This function just adds the matrices
    """
    if check_dim(mat1, mat2) == False:
        return None
    else:
        if isinstance(mat1[0], list):
            return [add_matrices(sub_mat1, sub_mat2)
                    for sub_mat1, sub_mat2 in zip(mat1, mat2)]
        else:
            return [val1 + val2 for val1, val2 in zip(mat1, mat2)]
        