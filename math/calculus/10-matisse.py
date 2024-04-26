#!/usr/bin/env python3
"""
This is some documenation
"""


def poly_derivative(poly):
    """
    Derivative of a polynomial
    """
    if not isinstance(poly, list)\
       or len(poly) == 0:
        return None
    else:
        d_x = []
        for i in range(len(poly)):
            d_x.append(poly[i]*i)
        if d_x[0] == 0:
            d_x.pop(0)
        elif d_x == [0]:
            return [0]
        return d_x