#/usr/bin/env python3
"""_summary_
This is some documentation
"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """_summary_

    Args:
        images (_type_): _description_
        kernel (_type_): _description_

    Returns:
        _type_: _description_
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    out_h = h - kh + 1
    out_w = w - kw + 1

    convolved_images = np.zeros((m, out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            convolved_images[:, i, j] = np.sum(images[:, i:i+kh, j:j+kw] * kernel, axis=(1, 2))

    return convolved_images
    