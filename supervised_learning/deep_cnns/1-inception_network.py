#!/usr/bin/env python3
"""
This is some documentation
"""
from tensorflow import jeras as K
inception_block = __import__(0)


def inception_block(A_prev, filters):
    F1, F3R, F3, F5R, F5, FPP = filters

    branch1 = Conv2D(F1, (1, 1), padding='same', activation='relu')(A_prev)

    branch2 = Conv2D(F3R, (1, 1), padding='same', activation='relu')(A_prev)
    branch2 = Conv2D(F3, (3, 3), padding='same', activation='relu')(branch2)

    branch3 = Conv2D(F5R, (1, 1), padding='same', activation='relu')(A_prev)
    branch3 = Conv2D(F5, (5, 5), padding='same', activation='relu')(branch3)

    branch4 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(A_prev)
    branch4 = Conv2D(FPP, (1, 1), padding='same', activation='relu')(branch4)

    output = concatenate([branch1, branch2, branch3, branch4], axis=-1)

    return output
