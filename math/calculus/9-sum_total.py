#!/usr/bin/env python3
"""
This module has only one method
"""


def summation_i_squared(n):
    """
    This method does is the total sum of n squares
    """
    if not isinstance(n, int) or n <= 0:
        return None
    return n * (n + 1) * (2 * n + 1) // 6
