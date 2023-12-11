#!/usr/bin/env python3
"""
This module plots two different rates of decay. This time for C-14 and Ra-226
"""
import numpy as np
import matplotlib.pyplot as plt


x = np.arange(0, 21000, 1000)
r = np.log(0.5)
t1 = 5730
t2 = 1600
y1 = np.exp((r / t1) * x)
y2 = np.exp((r / t2) * x)

plt.title('Exponential Decay of Radioactive Elements')
plt.xlim([0, 20000])
plt.xlabel('Time (years)')
plt.ylim([0, 1])
plt.ylabel('Fraction Remaining')
plt.plot(x, y1, 'r--', x, y2, 'g-')
plt.legend(['C-14', 'Ra-226'])
