#!/usr/bin/env python3
"""
This module defines a Binomial distribution
"""
pi = 3.1415926536
e = 2.7182818285


class Binomial:
    """
    This is a Binomial distribution
    """
    def __init__(self, data=None, n=1, p=0.5):
        if data is None:
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                mean = sum(data) / len(data)
                squares = [(x - mean) ** 2 for x in data]
                var = sum(squares) / len(squares)
                self.p = 1 - (var / mean)
                self.n = round(mean / self.p)
                self.p = mean / self.n
        if self.n <= 0:
            raise ValueError("n must be a positive value")
        if not 0 < self.p < 1:
            raise ValueError("p must be greater than 0 and less than 1")

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
            return k * Binomial.fact(k-1)

    def pmf(self, k):
        """
        This method is the probability mass
        function on a Binomial distribution
        """
        if not isinstance(k, int):
            k = int(k)
        if not 0 <= k <= self.n:
            return 0
        else:
            choose = (Binomial.fact(self.n) /
                      (Binomial.fact(k) * Binomial.fact(self.n - k)))
            return choose * (self.p**k) * (1 - self.p)**(self.n - k)

    def cdf(self, k):
        """
        This method is the cumulative distribution
        function on a Binomial distribution
        """
        if not isinstance(k, int):
            k = int(k)
        if not 0 <= k <= self.n:
            return 0
        else:
            cdf_value = 0
            for r in range(k + 1):
                cdf_value += self.pmf(r)
            return cdf_value
