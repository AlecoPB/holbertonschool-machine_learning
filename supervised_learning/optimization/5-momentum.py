#!/usr/bin/env python3
"""_summary_
This is some documentation
"""
import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """_summary_

    Args:
        alpha (_type_): _description_
        beta1 (_type_): _description_
        var (_type_): _description_
        grad (_type_): _description_
        v (_type_): _description_

    Returns:
        _type_: _description_
    """
    v = beta1 * v + (1 - beta1) * grad
    var = var - alpha * v
    return var, v
