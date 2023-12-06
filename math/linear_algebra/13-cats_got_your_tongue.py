#!/usr/bin/env python3
"""
This module has only one method
"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    This method concatenates two matrices along an axis
    """
    np.concatenate((mat1, mat2), axis = axis)
