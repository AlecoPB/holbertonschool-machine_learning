#!/usr/bin/env python3

def minor(matrix):
    """
    Calculate the minor matrix of a square matrix.
    """
    # Check if matrix is a list of lists
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    
    # Check if matrix is empty or not square
    n = len(matrix)
    if n == 0 or any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")
    
    # Function to calculate the determinant of a matrix
    def determinant(matrix):
        # Base cases
        if len(matrix) == 1:
            return matrix[0][0]
        if len(matrix) == 2:
            return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
        # Recursive case
        det = 0
        for c in range(len(matrix)):
            minor = [row[:c] + row[c+1:] for row in (matrix[1:])]
            det += ((-1) ** c) * matrix[0][c] * determinant(minor)
        return det
    
    # Calculate the minor matrix
    minor_matrix = []
    for i in range(n):
        minor_row = []
        for j in range(n):
            sub_matrix = [row[:j] + row[j+1:] for row in (matrix[:i] + matrix[i+1:])]
            minor_row.append(determinant(sub_matrix))
        minor_matrix.append(minor_row)
    
    return minor_matrix
