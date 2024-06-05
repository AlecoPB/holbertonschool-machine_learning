#!/usr/bin/env python3
"""
This is some documentation
"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Forward propagation for a convolutional layer.
    Returns:
        numpy.ndarray: Output of the convolutional layer.
    """
    # Extract dimensions
    (m, h_prev, w_prev, c_prev) = A_prev.shape
    (kh, kw, c_prev, c_new) = W.shape
    (sh, sw) = stride
    
    # Determine padding
    if padding == "same":
        ph = int(((h_prev - 1) * sh + kh - h_prev) / 2)
        pw = int(((w_prev - 1) * sw + kw - w_prev) / 2)
    elif padding == "valid":
        ph = 0
        pw = 0

    # Apply padding
    A_prev_pad = np.pad(A_prev, ((0, 0),
                                 (ph, ph),
                                 (pw, pw),
                                 (0, 0)),
                        mode='constant',
                        constant_values=(0, 0))

    # Compute output dimensions
    h_new = int((h_prev + 2 * ph - kh) / sh) + 1
    w_new = int((w_prev + 2 * pw - kw) / sw) + 1

    # Initialize output Z
    Z = np.zeros((m, h_new, w_new, c_new))

    # Perform convolution
    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    h_start = h * sh
                    h_end = h_start + kh
                    w_start = w * sw
                    w_end = w_start + kw

                    A_slice = A_prev_pad[i, h_start:h_end, w_start:w_end, :]
                    Z[i, h, w, c] = np.sum(A_slice * W[:, :, :, c]) + b[0, 0, 0, c]
    
    # Apply activation function
    A = activation(Z)
    
    return A
