#!/usr/bin/env python3
"""
This module only has one (1) method
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
