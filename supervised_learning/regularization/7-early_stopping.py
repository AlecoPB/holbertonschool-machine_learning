#!/usr/bin/env python3

def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Determines if gradient descent should be stopped early.

    Args:
    cost (float): The current validation cost of the neural network.
    opt_cost (float): The lowest recorded validation cost of the neural network.
    threshold (float): The threshold used for early stopping.
    patience (int): The patience count used for early stopping.
    count (int): The count of how long the threshold has not been met.

    Returns:
    bool: Whether the network should be stopped early.
    int: The updated count.
    """

    if cost > opt_cost - threshold:
        count += 1
        if count >= patience:
            return True, count
    else:
        count = 0

    return False, count