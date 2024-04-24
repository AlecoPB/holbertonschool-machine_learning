#!/usr/bin/env python3
"""
This module plots an exponential function
"""
import matplotlib.pyplot as plt
import numpy as np


def line():
    y = np.arange(0, 11) ** 3
    x = np.arange(0, 11)
    plt.plot(x, y, "r-")
    plt.xlim(0, 10)
    plt.show()
