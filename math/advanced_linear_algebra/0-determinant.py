#!/usr/bin/env python3

def determinant(matrix):
    """
    Calculate the determinant of a square matrix recursively.
    """
    # Check if matrix is a list of lists
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Check for 0x0 matrix
    if matrix == [[]]:
        return 1

    # Check if matrix is square
    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a square matrix")

    # Base case for 1x1 matrix
    if n == 1:
        return matrix[0][0]

    # Base case for 2x2 matrix
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    # Recursive case for matrix of size > 2
    det = 0
    for c in range(n):
        minor = [row[:c] + row[c+1:] for row in (matrix[1:])]
        det += ((-1) ** c) * matrix[0][c] * determinant(minor)
    return det
