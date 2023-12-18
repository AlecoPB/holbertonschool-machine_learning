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
                if round(mean / self.p) < 0:
                    raise ValueError("n must be a non-negative value")
                else:
                    self.n = round(mean / self.p)
                if not 0 <= mean / self.n <= 1:
                    raise ValueError("p must be greater than or equal to 0 and less than or equal to 1")
                else:
                    self.p = mean / self.n
