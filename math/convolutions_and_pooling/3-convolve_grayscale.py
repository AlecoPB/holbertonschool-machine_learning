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

    # Determine output shape based on padding
    if padding == 'same':
        pad_h = ((h - 1) * sh + kh - h) // 2 + 1
        pad_w = ((w - 1) * sw + kw - w) // 2 + 1
    elif padding == 'valid':
        pad_h, pad_w = 0, 0
    else:
        pad_h, pad_w = padding

    # Pad images
    images_padded = np.pad(images, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode='constant')

    # Calculate output shape
    out_h = (h + 2 * pad_h - kh) // sh + 1
    out_w = (w + 2 * pad_w - kw) // sw + 1

    # Initialize output
    convolved_images = np.zeros((m, out_h, out_w))

    # Perform convolution
    for i in range(out_h):
        for j in range(out_w):
            # Extract the region of interest from the padded image
            image_region = images_padded[:, i * sh:i * sh + kh, j * sw:j * sw + kw]
            # Perform element-wise multiplication and sum
            convolved_images[:, i, j] = np.sum(image_region * kernel, axis=(1, 2))

    return convolved_images
