#!/usr/bin/env python3
"""
This is some documentation
"""


class Poisson:
    """Class that represents a Poisson distribution."""

    e = 2.7182818285

    def _factorial(self, n):
        """
        Factorial function
        """
        if n == 0:
            return 1
        result = 1
        for i in range(1, n + 1):
            result *= i
        return result

    def __init__(self, data=None, lambtha=1.):
        """Initialize a Poisson distribution instance.

        Args:
            data (list, optional): Data used to estimate the distribution.
            lambtha (float): Expected number of occurrences in a given time frame.

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
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """Calculate the PMF for a given number of successes k.

        Args:
            k (int): The number of successes.

        Returns:
            float: The PMF value for k.
        """
        k = int(k)  # Convert k to an integer if not already
        if k < 0:
            return 0

        e = Poisson.e
        return (e ** -self.lambtha) * (self.lambtha ** k) / self._factorial(k)

    def cdf(self, k):
        """Calculate the CDF for a given number of successes k.

        Args:
            k (int): The number of successes.

        Returns:
            float: The CDF value for k.
        """
        k = int(k)  # Convert k to an integer if not already
        if k < 0:
            return 0

        cdf_value = 0
        for i in range(k + 1):
            cdf_value += self.pmf(i)
        return cdf_value
