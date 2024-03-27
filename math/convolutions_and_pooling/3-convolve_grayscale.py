#!/usr/bin/env python3
"""_summary_
This is some documentation
"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """_summary_

    Args:
        images (_type_): _description_
        kernel (_type_): _description_
        padding (str, optional): _description_. Defaults to 'same'.
        stride (tuple, optional): _description_. Defaults to (1, 1).

    Returns:
        _type_: _description_
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    if padding == 'valid':
        ph, pw = 0, 0
        out_h = (h - kh) // sh + 1
        out_w = (w - kw) // sw + 1
    elif padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2
        pw = ((w - 1) * sw + kw - w) // 2
        out_h = h
        out_w = w
    else:
        ph, pw = padding
        out_h = (h + 2 * ph - kh) // sh + 1
        out_w = (w + 2 * pw - kw) // sw + 1

    output = np.zeros((m, out_h, out_w))

    padded_images = np.pad(images, ((0,0), (ph,ph), (pw,pw)), mode='constant')

    for i in range(0, h - kh + 1, sh):
        for j in range(0, w - kw + 1, sw):
            output[:, i // sh, j // sw] = np.sum(
                padded_images[:, i:i+kh, j:j+kw] * kernel,
                axis=(1, 2)
            )

    return output
