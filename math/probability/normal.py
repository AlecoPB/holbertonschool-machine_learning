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
    def erf(x, terms=100):
        return (2/Normal.exp(1, terms)**0.5) * sum(((-1)**i * x**(2*i + 1))/(2*i + 1) for i in range(terms))

    def cdf(self, x):
        """
        Returns de CDF of a given x value
        """
        return 0.5 * (1 + self.erf((x - self.mean) / (self.stddev * (2 ** 0.5))))
