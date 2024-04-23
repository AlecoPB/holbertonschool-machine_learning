#!/usr/bin/env python3
"""
This is some documentation
"""


def cat_matrices(mat1, mat2, axis=0):
  """
  Concatenates two matrices (ndarrays) along a specific axis.

  Args:
      mat1: First matrix (ndarray).
      mat2: Second matrix (ndarray).
      axis: Axis along which to concatenate (default: 0).

  Returns:
      A new matrix (ndarray) containing the concatenated matrices, 
      or None if concatenation is not possible.
  """

  # Check if matrices are empty (given assumption)
  if not mat1.size or not mat2.size:
    return None

  # Validate data type consistency
  if mat1.dtype != mat2.dtype:
    return None

  # Validate dimension compatibility (except for concatenation axis)
  for i in range(len(mat1.shape)):
    if i != axis and mat1.shape[i] != mat2.shape[i]:
      return None

  # Concatenate based on axis
  if axis == 0:
    return np.concatenate((mat1, mat2))
  elif axis == 1:
    return np.concatenate((mat1, mat2), axis=axis)
  else:
    # Invalid axis provided
    return None