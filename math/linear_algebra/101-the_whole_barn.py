#!/usr/bin/env python3
"""
This module has only one method, unless they make me use a second method
"""


def matrix_shape(matrix):
  """
  Returns the shape of a matrix.
  """
  if not matrix:
    return []
  if isinstance(matrix[0], list):
    return [len(matrix)] + matrix_shape(matrix[0])
  else:
    return [len(matrix)]

def add_matrices(mat1, mat2):
    """
    This method adds up two ndmatrices
    """
    if matrix_shape(mat1) != matrix_shape(mat2):
       return None
    if isinstance(mat1[0], list):
       return [add_matrices(m1, m2) for m1, m2 in zip(mat1, mat2)]
    else:
       return [m1 + m2 for m1, m2 in zip(mat1, mat2)]
