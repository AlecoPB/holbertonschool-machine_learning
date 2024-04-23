#!/usr/bin/env python3
"""
This is some documentation
"""


def cat_matrices(mat1, mat2, axis=0):
    # Base case: if the matrices are 1-D lists, concatenate them
    if not isinstance(mat1[0], list):
        if axis == 0:
            return mat1 + mat2
        else:
            return None

    # Recursive case: if the matrices have multiple dimensions
    else:
        # Check if the matrices can be concatenated along the given axis
        if len(mat1) != len(mat2) and axis != 0:
            return None
        if len(mat1[0]) != len(mat2[0]) and axis != 1:
            return None

        # Concatenate the matrices
        result = []
        for m1, m2 in zip(mat1, mat2):
            concatenated = cat_matrices(m1, m2, axis - 1)
            if concatenated is None:
                return None
            result.append(concatenated)
        return result