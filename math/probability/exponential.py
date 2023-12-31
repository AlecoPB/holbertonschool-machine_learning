#!/usr/bin/env python3
"""
This module defines a class containing Exponential distribution
"""
pi = 3.1415926536
e = 2.7182818285


class Exponential:
    """
    This is the exponential distribution class
    """
    def __init__(self, data=None, lambtha=1.):
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            else:
                self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                self.lambtha = 1 / (sum(data) / len(data))

    @staticmethod
    def fact(k):
        """
        Just a factorial method
        """
        if not isinstance(k, int):
            raise TypeError("the number must be an integer")
        elif k == 0:
            return 1
        if k == 1:
            return k
        else:
            return k * Exponential.fact(k-1)

    def pdf(self, k):
        """
        This calculates de probability Density Function (PMF)
        """
        if k < 0:
            return 0
        return self.lambtha*(e**(-1*self.lambtha*k))

    def cdf(self, k):
        """
        This calculates the Cumulative Distribution Function (CDF)
        """
        if k < 0:
            return 0
        return 1 - e**(-1*self.lambtha*k)
