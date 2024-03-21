#!/usr/bin/env python3
import numpy as np

def tanh_activation(Z):
    return np.tanh(Z)

def softmax_activation(Z):
    expZ = np.exp(Z - np.max(Z))
    return expZ / expZ.sum(axis=0, keepdims=True)

def dropout_forward_prop(X, weights, L, keep_prob):
    cache = {}
    cache["A0"] = X
    m = X.shape[1]

    for l in range(1, L):
        Z = np.dot(weights['W' + str(l)], cache['A' + str(l - 1)]) + weights['b' + str(l)]
        A = tanh_activation(Z)
        D = np.random.rand(A.shape[0], A.shape[1])
        D = D < keep_prob
        A = np.multiply(A, D)
        A = A / keep_prob
        cache['Z' + str(l)] = Z
        cache['D' + str(l)] = D
        cache['A' + str(l)] = A

    Z = np.dot(weights['W' + str(L)], cache['A' + str(L - 1)]) + weights['b' + str(L)]
    A = softmax_activation(Z)
    cache['Z' + str(L)] = Z
    cache['A' + str(L)] = A

    return cache