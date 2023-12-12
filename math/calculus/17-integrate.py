#!/usr/bin/env python3
"""
This module has only one method
"""


def poly_integral(poly, C=0):
    """
    This method integrates a polynomial expression
    """
    int_poly = [C]
    if not isinstance(C, int) or not isinstance(poly, list) or len(poly) == 0:
        return None
    if poly == [0]:
        return [C]
    for coeff in range(len(poly)):
        if not isinstance(poly[coeff], (int, float)):
            return None
        api = poly[coeff] / (coeff + 1)
        if api - int(api) == 0:
            int_poly.append(int(api))
        else:
            int_poly.append(api)
    return int_poly
