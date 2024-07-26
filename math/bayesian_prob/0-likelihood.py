#!/usr/bin/env python3
"""
This module contains a function to calculate the likelihood of obtaining
data given various hypothetical probabilities of developing severe side effects.
"""


import numpy as np

def likelihood(x, n, P):
	"""
	Calculate the likelihood of obtaining the
    data given various hypothetical probabilities."""
	# Validate inputs
	if not isinstance(n, int) or n <= 0:
		raise ValueError("n must be a positive integer")
	if not isinstance(x, int) or x < 0:
		raise ValueError("x must be an integer that is greater than or equal to 0")
	if x > n:
		raise ValueError("x cannot be greater than n")
	if not isinstance(P, np.ndarray) or P.ndim != 1:
		raise TypeError("P must be a 1D numpy.ndarray")
	if np.any((P < 0) | (P > 1)):
		raise ValueError("All values in P must be in the range [0, 1]")

	# Calculate the binomial coefficient (n choose x)
	def binomial_coefficient(n, k):
		if k > n:
			return 0
		if k == 0 or k == n:
			return 1
		k = min(k, n - k)  # Take advantage of symmetry
		c = 1
		for i in range(k):
			c = c * (n - i) // (i + 1)
		return c

	# Calculate the likelihood for each probability in P
	binom_coeff = float(binomial_coefficient(n, x))
	likelihoods = binom_coeff * (P ** x) * ((1 - P) ** (n - x))

	return likelihoods
