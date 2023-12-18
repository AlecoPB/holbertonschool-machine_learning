#!/usr/bin/env python3
class Binomial:
    """
    This module defines a Binomial distribution
    """
    pi = 3.1415926536
    e = 2.7182818285
    
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
                self.n = mean / self.p
                self.p = self.p / self.n
