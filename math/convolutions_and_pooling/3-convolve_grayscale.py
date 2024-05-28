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

    if padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2 + 1
        pw = ((w - 1) * sw + kw - w) // 2 + 1
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    images_padded = np.pad(images,
                           ((0, 0),
                            (ph, ph),
                            (pw, pw)),
                           mode='constant')

    out_h = (h + 2 * ph - kh) // sh + 1
    out_w = (w + 2 * pw - kw) // sw + 1

    convolved_images = np.zeros((m, out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            image_region = images_padded[:,
                                         i * sh:i * sh + kh,
                                         j * sw:j * sw + kw]
            convolved_images[:, i, j] =\
                np.sum(image_region * kernel,
                       axis=(1, 2))

    return convolved_images
