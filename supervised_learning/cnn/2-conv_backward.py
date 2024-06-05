#!/usr/bin/env python3
"""
This is some documentation
"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    - dA_prev: Partial derivatives with respect to the previous layer
    - dW: Partial derivatives with respect to the kernels
    - db: Partial derivatives with respect to the biases
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride
    _, h_new, w_new, _ = dZ.shape

    if padding == "same":
        ph = ((h_prev - 1) * sh + kh - h_prev) // 2 + 1
        pw = ((w_prev - 1) * sw + kw - w_prev) // 2 + 1
    else:
        ph = 0
        pw = 0

    A_prev_padded = np.pad(A_prev, ((0, 0), (ph, ph),
                                    (pw, pw), (0, 0)),
                           mode='constant',
                           constant_values=0)
    dA_prev_padded = np.zeros_like(A_prev_padded)
    dW = np.zeros_like(W)
    db = np.zeros_like(b)

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    A_slice = A_prev_padded[i,
                                            vert_start:vert_end,
                                            horiz_start:horiz_end, :]

                    dA_prev_padded[i, vert_start:vert_end,
                                   horiz_start:horiz_end,
                                   :] += W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += A_slice * dZ[i, h, w, c]
                    db[:, :, :, c] += dZ[i, h, w, c]

    if padding == "same":
        dA_prev = dA_prev_padded[:, ph:-ph, pw:-pw, :]
    else:
        dA_prev = dA_prev_padded

    return dA_prev, dW, db
