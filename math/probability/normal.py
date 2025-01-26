#!/usr/bin/env python3
"""
This is some documentation
"""


class Normal:
    """Class that represents a normal distribution."""

    def __init__(self, data=None, mean=0., stddev=1.):
        """Initialize a Normal distribution instance.

        Args:
            data (list, optional): Data used to estimate the distribution.
            mean (float): The mean of the distribution.
            stddev (float): The standard deviation of the distribution.

        Raises:
            TypeError: If data is not a list.
            ValueError: If data contains fewer than two data points.
            ValueError: If stddev is not a positive value.
        """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = sum(data) / len(data)
            variance = sum((x - self.mean) ** 2 for x in data) / len(data)
            self.stddev = variance ** 0.5

    def z_score(self, x):
        """Calculate the z-score of a given x-value.

        Args:
            x (float): The x-value.

        Returns:
            float: The z-score of x.
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Calculate the x-value of a given z-score.

        Args:
            z (float): The z-score.

        Returns:
            float: The x-value corresponding to z.
        """
        return z * self.stddev + self.mean

    def pdf(self, x):
        """Calculate the PDF for a given x-value.

        Args:
            x (float): The x-value.

        Returns:
            float: The PDF value for x.
        """
        return ((1 / (self.stddev * (2 * 3.1415926536) ** 0.5))
                * (2.7182818285 ** (-0.5 * ((x - self.mean)
                                            / self.stddev) ** 2)))

    def cdf(self, x):
        """Calculate the CDF for a given x-value.

        Args:
            x (float): The x-value.

        Returns:
            float: The CDF value for x.
        """
        return 1 / 2 * (1 + self.erf((x - self.mean)
                        / (self.stddev * pow(2, 1/2))))