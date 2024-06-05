#!/usr/bin/env python3
"""
This is some documentation
"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Returns:
    - The output of the pooling layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    h_new = (h_prev - kh) // sh + 1
    w_new = (w_prev - kw) // sw + 1

    A = np.zeros((m, h_new, w_new, c_prev))

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_prev):
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    A_slice = A_prev[i, vert_start:vert_end,
                                     horiz_start:horiz_end, c]

                    if mode == 'max':
                        A[i, h, w, c] = np.max(A_slice)
                    elif mode == 'avg':
                        A[i, h, w, c] = np.mean(A_slice)

    return A
