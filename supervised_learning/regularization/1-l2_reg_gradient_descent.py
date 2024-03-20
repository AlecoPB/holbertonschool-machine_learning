#!/usr/bin/env python3
"""_summary_
This is some documentation
"""
import numpy as np

def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    m = Y.shape[1]
    weights_copy = weights.copy()
    dz = cache['A' + str(L)] - Y

    for i in range(L, 0, -1):
        A_prev = cache['A' + str(i-1)]
        dw = (1/m) * np.dot(dz, A_prev.T) + ((lambtha/m) * weights['W' + str(i)])
        db = (1/m) * np.sum(dz, axis=1, keepdims=True)
        if i > 1:
            da = 1 - np.square(A_prev)
            dz = np.dot(weights_copy['W' + str(i)].T, dz) * da
        weights['W' + str(i)] -= alpha * dw
        weights['b' + str(i)] -= alpha * db

    return weights
