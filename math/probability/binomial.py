#!/usr/bin/env python3
"""
This is some documentation
"""


class Binomial:
    """Class that represents a binomial distribution."""

    def __init__(self, data=None, n=1, p=0.5):
        """Initialize a Binomial distribution instance.

        Args:
            data (list, optional): Data used to estimate the distribution.
            n (int): Number of Bernoulli trials.
            p (float): Probability of success.

        Raises:
            TypeError: If data is not a list.
            ValueError: If data contains fewer than two data points.
            ValueError: If n is not a positive value.
            ValueError: If p is not a valid probability.
        """
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)
            self.p = 1 - (variance / mean)
            self.n = round(mean / self.p)
            self.p = mean / self.n

    def _factorial(self, num):
        """Calculate the factorial of a number."""
        if num == 0:
            return 1
        result = 1
        for i in range(1, num + 1):
            result *= i
        return result

    def pmf(self, k):
        """Calculate the PMF for a given number of successes k.

        Args:
            k (int): The number of successes.

        Returns:
            float: The PMF value for k.
        """
        k = int(k)  # Convert k to an integer if not already
        if k < 0 or k > self.n:
            return 0

        combination = (self._factorial(self.n)
                       / (self._factorial(k)
                          * self._factorial(self.n - k)))
        return combination * (self.p ** k) * ((1 - self.p) ** (self.n - k))

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
