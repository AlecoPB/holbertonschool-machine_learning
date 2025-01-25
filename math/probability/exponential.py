#!/usr/bin/env python3
"""
This is some documentation
"""


class Exponential:
    """Class that represents an exponential distribution."""

    def __init__(self, data=None, lambtha=1.):
        """Initialize an Exponential distribution instance.

        Args:
            data (list, optional): Data used to estimate the distribution.
            lambtha (float): Expected number of occurrences in a given time
            frame.

        Raises:
            TypeError: If data is not a list.
            ValueError: If data contains fewer than two data points.
            ValueError: If lambtha is not a positive value.
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = 1 / (sum(data) / len(data))
