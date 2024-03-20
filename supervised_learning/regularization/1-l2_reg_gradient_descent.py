#!/usr/bin/env python3
"""_summary_
This is some documentation
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """_summary_

    Args:
        Y (_type_): _description_
        weights (_type_): _description_
        cache (_type_): _description_
        alpha (_type_): _description_
        lambtha (_type_): _description_
        L (_type_): _description_

    Returns:
        _type_: _description_
    """
    m = Y.shape[1]
    weights_copy = weights.copy()
    dz = cache['A' + str(L)] - Y

    for i in range(L, 0, -1):
        A_prev = cache['A' + str(i-1)]
        db = (1/m) * np.sum(dz, axis=1, keepdims=True)
        dw = (1/m) * np.dot(dz, A_prev.T) +\
            ((lambtha/m) * weights['W' + str(i)])
        if i > 1:
            da = 1 - np.square(A_prev)
            dz = np.dot(weights_copy['W' + str(i)].T, dz) * da
        weights['W' + str(i)] -= alpha * dw
        weights['b' + str(i)] -= alpha * db

    return weights
