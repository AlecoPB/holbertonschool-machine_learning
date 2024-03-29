#!/usr/bin/env python3
"""_summary_

Returns:
    _type_: _description_
"""
import numpy as np


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Determines if gradient descent should be stopped early.

    Parameters:
    cost (float): The current validation cost of the neural network.
    opt_cost (float): The lowest recorded validation
    cost of the neural network.
    threshold (float): The threshold used for early stopping.
    patience (int): The patience count used for early stopping.
    count (int): The count of how long the threshold has not been met.

    Returns:
    bool: Whether the network should be stopped early.
    int: The updated count.
    """
    if opt_cost - cost < threshold:
        # print("Se sumó")
        count += 1
    else:
        # print("Se seteó")
        count = 0

    # Check if the count exceeds the patience
    if count >= patience:
        return True, count

    return False, count
