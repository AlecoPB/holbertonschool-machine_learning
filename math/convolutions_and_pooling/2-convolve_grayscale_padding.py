#!/usr/bin/env python3
"""_summary_

Returns:
    _type_: _description_
"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """_summary_

    Args:
        images (_type_): _description_
        kernel (_type_): _description_

    Returns:
        _type_: _description_
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    out_h = h + 2 * ph - kh + 1
    out_w = w + 2 * pw - kw + 1

    output = np.zeros((m, out_h, out_w))

    padded_images = np.pad(
        images,
        ((0, 0),
         (ph, ph),
         (pw, pw)),
        mode='constant')

    for i in range(out_h):
        for j in range(out_w):
            output[:, i, j] =\
                np.sum(padded_images[:, i:i+kh, j:j+kw] * kernel,
                       axis=(1, 2))

    return output
