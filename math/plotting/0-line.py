#!/usr/bin/env python3
"""
This is some documentation
"""
import numpy as np
import matplotlib.pyplot as plt

def line():
    """
    Basic line plotting
    """
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))
    plt.plot(np.arange(0, 11), y, 'r-')
    plt.xlim(0, 10)
    plt.show()
