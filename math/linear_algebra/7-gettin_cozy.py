#!/usr/bin/env python3
"""
This module has only one method
"""
cat_arrays = __import__('6-howdy_partner').cat_arrays


def cat_matrices2D(mat1, mat2, axis=0):
    """
    This method concatenates two matrices
    """
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        return mat1 + mat2
    elif axis == 1:
        if len(mat1) != len(mat2):
            return None
        cat_mat = []
        for i in range(len(mat1)):
            cat_mat.append(cat_arrays(mat1[i], mat2[i]))
        return cat_mat
