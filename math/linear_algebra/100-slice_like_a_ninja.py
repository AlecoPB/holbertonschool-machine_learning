#!/usr/bin/env python3
"""
This is some documentation
"""


def np_slice(matrix, axes={}):
    new_arr = []
    for key, value in axes.items():
        val1 = value[0]
        val2 = value[1] if len(value) > 1 else None
        if matrix.ndim == 1:
            return matrix[val1:val2]
        else:
            new_axes = {k: v for k, v in axes.items() if k >= key}
            for i in range(len(matrix)):
                temp_arr = np_slice(matrix[i], new_axes)
                new_arr.append([])
                for j in temp_arr:
                    new_arr[i].append(j)
    return new_arr
