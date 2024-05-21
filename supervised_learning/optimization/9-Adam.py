#!/usr/bin/env python3
"""_summary_
This is some documentation
"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    Updates a variable in place using the Adam optimization algorithm.

    Args:
        alpha: The learning rate.
        beta1: Weight for the first moment (momentum).
        beta2: Weight for the second moment (RMSProp).
        epsilon: A small number to avoid division by zero.
        var: Numpy array containing the variable to be updated.
        grad: Numpy array containing the gradient of var.
        v: Previous first moment (initialized as zeros).
        s: Previous second moment (initialized as zeros).
        t: Time step used for bias correction.

    Returns:
        Updated variable, new first moment, and new second moment.
    """
    # Update biased first moment estimate
    v = beta1 * v + (1 - beta1) * grad

    # Update biased second moment estimate
    s = beta2 * s + (1 - beta2) * np.square(grad)

    # Bias correction
    v_corrected = v / (1 - beta1**t)
    s_corrected = s / (1 - beta2**t)

    # Update variable
    var -= alpha * v_corrected / (np.sqrt(s_corrected) + epsilon)

    return var, v, s
