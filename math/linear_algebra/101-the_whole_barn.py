#!/usr/bin/env python3
"""
This is some documentation
"""

def check_dim(mat1, mat2):
    """
    This function checks if the dimensions of
    both matrices are the same
    """
    if mat1.ndim == 1:
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
        return mat1 + mat2
