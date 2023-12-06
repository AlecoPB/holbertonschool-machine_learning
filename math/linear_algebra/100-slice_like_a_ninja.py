#!/usr/bin/env python3
"""
This module has only one method
"""
import numpy as np


def np_slice(matrix, axes={}):
    """
    This method slices a matrix along the axes that are specified following the next structure
      axes = {Start:Stop, Step)
    """
    lil_slice = []
    for i in range(len(matrix)):
        if i in axes:
          lil_slice.append(slice(*axes[i]))
        else:
          lil_slice.append(slice(None))
    return matrix[tuple(lil_slice)]
