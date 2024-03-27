#!/usr/bin/env python3
"""_summary_
This is some documentation
"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """_summary_

    Args:
        images (_type_): _description_
        kernel (_type_): _description_
        padding (str, optional): _description_. Defaults to 'same'.
        stride (tuple, optional): _description_. Defaults to (1, 1).

    Returns:
        _type_: _description_
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    out_h = (h - kh) // sh + 1
    out_w = (w - kw) // sw + 1

    output = np.zeros((m, out_h, out_w, c))

    for i in range(out_h):
        for j in range(out_w):
            h_start = i * sh
            h_end = h_start + kh
            w_start = j * sw
            w_end = w_start + kw

            if mode == 'max':
                output[:, i, j, :] =\
                    np.max(images[:,
                                  h_start:h_end,
                                  w_start:w_end,
                                  :],
                           axis=(1, 2))
            elif mode == 'avg':
                output[:, i, j, :] =\
                    np.mean(images[:,
                                   h_start:h_end,
                                   w_start:w_end, :],
                            axis=(1, 2))
    return output
