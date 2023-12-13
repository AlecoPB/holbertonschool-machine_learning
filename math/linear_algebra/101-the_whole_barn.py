#!/usr/bin/env python3
"""
This module has only one method
"""
import numpy as np


def add_matrices(mat1, mat2):
    """
    This method adds up two ndmatrices
    """
    if np.shape(mat1) != np.shape(mat2):
       return None
    if isinstance(mat1[0], list):
       return [add_matrices(m1, m2) for m1, m2 in zip(mat1, mat2)]
    else:
       return [m1 + m2 for m1, m2 in zip(mat1, mat2)]
