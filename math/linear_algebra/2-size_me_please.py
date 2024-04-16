#!/usr/bin/env python3

def matrix_shape(matrix):
    length = []
    if not isinstance(matrix[0], list):
        length.append(len(matrix))
        return length
    else:
        length.append(len(matrix))
        prev_res = matrix_shape(matrix[0])
        for i in range(0, len(prev_res)):
            length.append(prev_res[i])
        return length
