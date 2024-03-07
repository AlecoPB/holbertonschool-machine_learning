#!/usr/bin/env python3
"""_summary_
This is some documentation
"""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Updates a variable using the RMSProp optimization algorithm.

    Args:
        alpha (float): Learning rate.
        beta2 (float): RMSProp weight.
        epsilon (float): Small number to avoid division by zero.
        var (numpy.ndarray): Variable to be updated.
        grad (numpy.ndarray): Gradient of var.
        s (numpy.ndarray): Previous second moment of var.

    Returns:
        tuple: Updated variable and new moment.
    """
    # Compute the squared gradient (element-wise)
    s_new = beta2 * s + (1 - beta2) * grad**2
    
    # Update the variable
    var_new = var - alpha * grad / (np.sqrt(s_new) + epsilon)
    
    return var_new, s_new
