#!/usr/bin/env python3
"""
This is some documentation
"""


def np_slice(matrix, axes={}):
    """
    This is some more docu
    """
    slices = []
    for axis in range(len(matrix.shape)):
        if axis in axes:
            slices.append(slice(*axes[axis]))
        else:
            slices.append(slice(None, None))
    sliced_matrix = matrix[tuple(slices)]
    return sliced_matrix
