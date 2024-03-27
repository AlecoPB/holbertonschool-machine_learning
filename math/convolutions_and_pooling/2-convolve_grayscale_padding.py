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

    padded_images = np.pad(
        images,
        ((0, 0),
         (ph, ph),
         (pw, pw)),
        mode='constant')

    convolved_images = np.zeros((m, h, w))

    for i in range(h):
        for j in range(w):
            convolved_images[:, i, j] =\
                np.sum(padded_images[:, i:i+kh, j:j+kw] * kernel,
                       axis=(1, 2))

    return convolved_images
