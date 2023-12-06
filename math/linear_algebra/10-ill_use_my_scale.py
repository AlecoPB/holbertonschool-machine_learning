#!/usr/bin/env python3
"""
This module serves almost no purpose
"""


def np_shape(matrix):
  """
  This method is literally just renaming np.shape
  """
  row_len = len(matrix)
  col_len = len(matrix[0])
  return (row_len, col_len)
