#!/usr/bin/env python3
"""
This is some documentation
"""
from tensorflow import keras as K


def inception_block(A_prev, filters):
    F1, F3R, F3, F5R, F5, FPP = filters

    conv_1x1 = K.Conv2D(F1, (1, 1), activation='relu', padding='same')(A_prev)

    conv_3x3_reduce = K.Conv2D(F3R, (1, 1), activation='relu', padding='same')(A_prev)
    conv_3x3 = K.Conv2D(F3, (3, 3), activation='relu', padding='same')(conv_3x3_reduce)

    conv_5x5_reduce = K.Conv2D(F5R, (1, 1), activation='relu', padding='same')(A_prev)
    conv_5x5 = K.Conv2D(F5, (5, 5), activation='relu', padding='same')(conv_5x5_reduce)

    max_pool = K.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(A_prev)
    max_pool_conv = K.Conv2D(FPP, (1, 1), activation='relu', padding='same')(max_pool)

    output = K.concatenate([conv_1x1, conv_3x3, conv_5x5, max_pool_conv], axis=-1)

    return output
