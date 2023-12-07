#!/usr/bin/env python3
"""
This module plots the decay of the C-14 isotope
"""
import numpy as np
import matplotlib.pyplot as plt


x = np.arange(0, 28651, 5730)
r = np.log(0.5)
t = 5730
y = np.exp((r / t) * x)

plt.plot(x, y)
plt.title('Exponential Decay of C-14')
plt.xlabel('Time (years)')
plt.xlim([0, 28651])
plt.yscale('log')
plt.ylabel('Fraction Remaining')
plt.show()
