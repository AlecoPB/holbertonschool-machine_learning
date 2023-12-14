#!/usr/bin/env python3
"""
This is a module containing a poisson distribution class
"""


class Poisson:
    """
    This is the poisson distribution class
    """
    def __init__(self, data=None, lambtha=1.):
        if data == None:
            if labtha <= 0:
                raise ValueError("lambtha must be a positive value")
            else:
                self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                self.lambtha = float(sum(data)/len(data))
