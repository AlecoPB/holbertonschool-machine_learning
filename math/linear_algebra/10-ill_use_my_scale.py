#!/usr/bin/env python3
"""
This module serves almost no purpose
"""


def np_shape(matrix):
  """
  This method is literally just renaming np.shape
  """
  return (np.shape(matrix) + np_shape(matrix[0]))
