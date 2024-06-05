#!/usr/bin/env python3
"""
This is some documentation
"""
import numpy as np

def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Forward propagation for a convolutional layer.

    Args:
        A_prev (numpy.ndarray): Output of the previous layer (shape: (m, h_prev, w_prev, c_prev)).
        W (numpy.ndarray): Kernels for the convolution (shape: (kh, kw, c_prev, c_new)).
        b (numpy.ndarray): Biases applied to the convolution (shape: (1, 1, 1, c_new)).
        activation (function): Activation function applied to the convolution.
        padding (str, optional): Type of padding used ("same" or "valid"). Defaults to "same".
        stride (tuple, optional): Strides for the convolution (sh, sw). Defaults to (1, 1).

    Returns:
        numpy.ndarray: Output of the convolutional layer.
    """
    # Get dimensions
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    if padding == "same":
        ph = max((h_prev - 1) * sh + kh - h_prev, 0)
        pw = max((w_prev - 1) * sw + kw - w_prev, 0)
    else:
        ph = 0
        pw = 0


    pad_top = ph // 2
    pad_bottom = ph - pad_top
    pad_left = pw // 2
    pad_right = pw - pad_left

    A_prev_padded = np.pad(A_prev, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant')
    
    h_new = (h_prev + 2 * ph - kh) // sh + 1
    w_new = (w_prev + 2 * pw - kw) // sw + 1
    
    Z = np.zeros((m, h_new, w_new, c_new))
    
    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw
                    
                    A_slice = A_prev_padded[i, vert_start:vert_end, horiz_start:horiz_end, :]
                    Z[i, h, w, c] = np.sum(A_slice * W[:, :, :, c]) + b[:, :, :, c]
    
    A = activation(Z)

    return A
