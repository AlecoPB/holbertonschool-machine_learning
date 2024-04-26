#!/usr/bin/env python3


def summation_i_squared(n):
    if n == 1:
        return n**2
    else:
        return n**2 + summation_i_squared(n-1)
