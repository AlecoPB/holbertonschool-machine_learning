#!/usr/bin/env python3
"""
This module has only one method
"""


def poly_derivative(poly):
    """
    This method derives polynomial expressions
    """
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    dev_poly = [0]
    if poly == [0]:
        return dev_poly
    for coeff in range(1, len(poly)):
        if coeff == 1:
            dev_poly[0] = poly[coeff] * coeff
        else:
            dev_poly.append(poly[coeff] * coeff)
    return dev_poly
