#!/usr/bin/env python3
"""
This is some documentation
"""


def early_stopping(cost, opt_cost, threshold,
                   patience, count):
    """
    Evaluates whether to stop gradient descent early.

    Parameters:
    - cost: The current validation cost of the neural network.
    - opt_cost: The lowest recorded validation cost of the neural network.
    - threshold: The threshold value used for determining early stopping.
    - patience: The number of iterations to wait before stopping.
    - count: The current count of iterations where the threshold
    has not been met.

    Returns: A tuple containing a boolean indicating if early stopping
    should occur, along with the updated count.
    """

    if (opt_cost - cost) <= threshold:
        count += 1
    else:
        count = 0

    return count >= patience, count
