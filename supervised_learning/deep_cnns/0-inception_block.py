#!/usr/bin/env python3
"""
This is some documentation
"""
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, concatenate


def inception_block(A_prev, filters):
    F1, F3R, F3, F5R, F5, FPP = filters

    conv_1x1 = Conv2D(F1, (1, 1), activation='relu', padding='same')(A_prev)

    conv_3x3_reduce = Conv2D(F3R, (1, 1), activation='relu', padding='same')(A_prev)
    conv_3x3 = Conv2D(F3, (3, 3), activation='relu', padding='same')(conv_3x3_reduce)

    conv_5x5_reduce = Conv2D(F5R, (1, 1), activation='relu', padding='same')(A_prev)
    conv_5x5 = Conv2D(F5, (5, 5), activation='relu', padding='same')(conv_5x5_reduce)

    max_pool = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(A_prev)
    max_pool_conv = Conv2D(FPP, (1, 1), activation='relu', padding='same')(max_pool)

    output = concatenate([conv_1x1, conv_3x3, conv_5x5, max_pool_conv], axis=-1)

    return output
