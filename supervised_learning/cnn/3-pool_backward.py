#!/usr/bin/env python3
"""
This is some documentation
"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Backwards propagation on a pooling neural network
    """
    (m, h_new, w_new, c_new) = dA.shape
    (h_prev, w_prev, c) = A_prev.shape[1:]
    (kh, kw) = kernel_shape
    (sh, sw) = stride
    
    dA_prev = np.zeros_like(A_prev)

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    h_start = h * sh
                    h_end = h_start + kh
                    w_start = w * sw
                    w_end = w_start + kw

                    if mode == 'max':
                        a_prev_slice = A_prev[i, h_start:h_end, w_start:w_end, c]
                        mask = (a_prev_slice == np.max(a_prev_slice))
                        dA_prev[i, h_start:h_end, w_start:w_end, c] += mask * dA[i, h, w, c]
                    elif mode == 'avg':
                        da = dA[i, h, w, c]
                        shape = (kh, kw)
                        average = da / (kh * kw)
                        dA_prev[i, h_start:h_end, w_start:w_end, c] += np.ones(shape) * average

    return dA_prev
