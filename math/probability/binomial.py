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
                self.n = round(max(data))
                self.p = sum(data) / (self.n * len(data))
