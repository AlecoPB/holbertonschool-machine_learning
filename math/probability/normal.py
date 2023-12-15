#!/usr/bin/env python3
"""
This module defines a class containing Normal Distribution
"""
pi = 3.1415926536
e = 2.7182818285


class Normal:
    """
    This is a Normal Distribution function
    """
    def __init__(self, data=None, mean=0., stddev=1):
        if data is None:
            self.mean = float(mean)
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            else:
                self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                self.mean = float(sum(data)/len(data))
                self.stddev = ((sum((i - self.mean)**2
                                    for i in data)/len(data))**0.5)

    def z_score(self, x):
        """
        Returns the z score of a given x value
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        Returns the x value of a given z score
        """
        return (z*self.stddev) + self.mean

    def pdf(self, x):
        """
        Returns the probability density function of a normal distribution on a given x value
        """
        return ((1 / (self.stddev * ((2 * pi) ** 0.5)))
                * (e ** (-1 * 0.5 * ((x - self.mean) / self.stddev) ** 2)))

    @staticmethod
    def factorial(n):
        if n == 0:
            return 1
        else:
            return n * Normal.factorial(n-1)

    @staticmethod
    def erf_taylor(x):
        """
        Compute the error function using a Taylor series.
        """
        sum_terms = 0.0
        for n in range(10):
            term = ((-1)**n * x**(2*n+1)) / (Normal.factorial(n) * (2*n+1))
            sum_terms += term
        return (2.0 / pi) * sum_terms

    def cdf(self, x):
        """Compute the CDF for a standard normal distribution."""
        z = self.z_score(x)
        return 0.5 * (1 + Normal.erf_taylor(z / (2**0.5)))
