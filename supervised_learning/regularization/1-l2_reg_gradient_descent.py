#!/usr/bin/env python3
"""_summary_
This is some documentation
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    m = Y.shape[1]
    dz_prev = cache['A' + str(L)] - Y

    for l in range(L, 0, -1):
        A_prev = cache['A' + str(l-1)]
        dw = (1/m) * np.dot(dz_prev, A_prev.T) + ((lambtha/m) * weights['W' + str(l)])
        db = (1/m) * np.sum(dz_prev, axis=1, keepdims=True)
        if l > 1:
            W_prev = weights['W' + str(l-1)]
            dz_prev = np.dot(W_prev.T, dz_prev) * (1 - cache['A' + str(l-1)] ** 2)

        weights['W' + str(l)] -= alpha * dw
        weights['b' + str(l)] -= alpha * db
