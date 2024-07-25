#!/usr/bin/env python3
"""
Tjis is some documentation
"""


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


def minor(matrix):
    """
    Calculate the minor matrix of a square matrix.
    """
    # Check if matrix is a list of lists
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Check if matrix is square and non-empty
    n = len(matrix)
    if n == 0 or any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    if len(matrix) == 1:
        return [[1]]

    # Calculate the minor matrix
    minor_matrix = []
    for i in range(len(matrix)):
        minor_row = []
        for j in range(len(matrix)):
            sub_matrix = [row[:j] + row[j+1:] for
                          row in (matrix[:i] + matrix[i+1:])]
            minor_row.append(determinant(sub_matrix))
        minor_matrix.append(minor_row)

    return minor_matrix


def cofactor(matrix):
    """
    Calculate the cofactor matrix of a square matrix.
    """
    # Check if matrix is a list of lists
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Check if matrix is square and non-empty
    n = len(matrix)
    if n == 0 or any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    if len(matrix) == 1:
        return [[1]]

    # Helper function to calculate the determinant
    def determinant(matrix):
        if len(matrix) == 1:
            return matrix[0][0]
        if len(matrix) == 2:
            return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
        det = 0
        for c in range(len(matrix)):
            sub_matrix = [row[:c] + row[c+1:] for row in (matrix[1:])]
            det += ((-1) ** c) * matrix[0][c] * determinant(sub_matrix)
        return det

    # Calculate the cofactor matrix
    cofactor_matrix = []
    for i in range(n):
        cofactor_row = []
        for j in range(n):
            sub_matrix = [row[:j] + row[j+1:]
                          for row in (matrix[:i] + matrix[i+1:])]
            cofactor_value = ((-1) ** (i + j)) * determinant(sub_matrix)
            cofactor_row.append(cofactor_value)
        cofactor_matrix.append(cofactor_row)

    return cofactor_matrix


def adjugate(matrix):
    """
    Calculate the adjugate matrix of a square matrix.
    """
    # Check if matrix is a list of lists
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Check if matrix is square and non-empty
    n = len(matrix)
    if n == 0 or any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    if len(matrix) == 1:
        return [[1]]

    # Helper function to calculate the determinant
    def determinant(matrix):
        if len(matrix) == 1:
            return matrix[0][0]
        if len(matrix) == 2:
            return (matrix[0][0] * matrix[1][1]
                    - matrix[0][1] * matrix[1][0])
        det = 0
        for c in range(len(matrix)):
            sub_matrix = [row[:c] + row[c+1:] for row in (matrix[1:])]
            det += ((-1) ** c) * matrix[0][c] * determinant(sub_matrix)
        return det

    # Calculate the cofactor matrix
    cofactor_matrix = []
    for i in range(n):
        cofactor_row = []
        for j in range(n):
            sub_matrix = [row[:j] + row[j+1:]
                          for row in (matrix[:i] + matrix[i+1:])]
            cofactor_value = ((-1) ** (i + j)) * determinant(sub_matrix)
            cofactor_row.append(cofactor_value)
        cofactor_matrix.append(cofactor_row)

    # Transpose the cofactor matrix to get the adjugate matrix
    adjugate_matrix = [[cofactor_matrix[j][i]
                        for j in range(n)] for i in range(n)]

    return adjugate_matrix
