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

    # Apply padding
    if padding == "same":
        pad_h = int(np.ceil((h_prev * sh - h_prev + kh - 1) / 2))
        pad_w = int(np.ceil((w_prev * sw - w_prev + kw - 1) / 2))
        A_prev_padded = np.pad(A_prev, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode="constant")
    else:
        A_prev_padded = A_prev

    # Compute output dimensions
    h_out = int((h_prev - kh + 2 * pad_h) / sh) + 1
    w_out = int((w_prev - kw + 2 * pad_w) / sw) + 1

    # Initialize output feature maps
    Z = np.zeros((m, h_out, w_out, c_new))

    # Convolution
    for i in range(h_out):
        for j in range(w_out):
            h_start, h_end = i * sh, i * sh + kh
            w_start, w_end = j * sw, j * sw + kw
            A_slice = A_prev_padded[:, h_start:h_end, w_start:w_end, :]
            Z[:, i, j, :] = np.sum(A_slice * W, axis=(1, 2, 3)) + b

    # Apply activation function
    A = activation(Z)

    return A