#!/usr/bin/env python3
"""
This is some documentation
"""


def np_elementwise(mat1, mat2):
    """_summary_

    Args:
        mat1 (_type_): _description_
        mat2 (_type_): _description_

    Returns:
        _type_: _description_
    """
    addition = mat1 + mat2
    substraction = mat1 - mat2
    multiplication = mat1 * mat2
    division = mat1 / mat2
    return (addition, substraction, multiplication, division,)
