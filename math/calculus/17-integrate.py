#!/usr/bin/env python3
"""
This is some documentation
"""


def poly_integral(poly, C=0):
    """_summary_

    Args:
        poly (_type_): _description_
        C (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    if not isinstance(poly, list) or\
       not all(isinstance(coef, (int, float)) for coef in poly) or\
       not isinstance(C, int) or\
       len(poly) == 0:
        return None

    integral = [C]
    for i, coef in enumerate(poly):
        if coef == 0:
            integral.append(0)
        else:
            integral_coef = coef / (i + 1)
            integral.append(int(integral_coef) if integral_coef.is_integer() else integral_coef)

    # Remove trailing zeros
    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    return integral
