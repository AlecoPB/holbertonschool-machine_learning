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
            # For 2D matrices, iterate over the rows and columns
            return [[mat1[i][j] + mat2[i][j] for j in range(len(mat1[0]))] for i in range(len(mat1))]
        else:
            # For 1D matrices, iterate over the elements
            return [mat1[i] + mat2[i] for i in range(len(mat1))]
