#!/usr/bin/env python3
"""
This is some documentation
"""
import numpy as np
import matplotlib.pyplot as plt

def change_scale():
    """
    Basic line plot that shows the
    exponential decay of C-14 
    """
    x = np.arange(0, 28651, 5730)
    r = np.log(0.5)
    t = 5730
    y = np.exp((r / t) * x)
    plt.figure(figsize=(6.4, 4.8))

    plt.plot(x, y)
    plt.title("Exponential Decay of C-14")
    plt.xlabel('Time (years)')
    plt.xlim([0, 28651])
    plt.yscale('log')
    plt.ylabel('Fraction Remaining')
    plt.show()
