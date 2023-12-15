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
            self.mean = mean
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            else:
                self.stddev = stddev
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                self.mean = sum(data)/len(data)
                self.stddev = (sum((i - self.mean)**2 for i in data)/len(data))**0.5
