#!/usr/bin/env python3
"""
This is some documentation
"""
import numpy as np
import matplotlib.pyplot as plt


def all_in_one():
    """
    Figure with 5 sub-plots
    """
    y0 = np.arange(0, 11) ** 3

    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
    y1 += 180

    x2 = np.arange(0, 28651, 5730)
    r2 = np.log(0.5)
    t2 = 5730
    y2 = np.exp((r2 / t2) * x2)

    x3 = np.arange(0, 21000, 1000)
    r3 = np.log(0.5)
    t31 = 5730
    t32 = 1600
    y31 = np.exp((r3 / t31) * x3)
    y32 = np.exp((r3 / t32) * x3)

    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)

    fig = plt.figure(figsize=(9, 8))

    axs1 = fig.add_subplot(3, 2, 1)
    axs2 = fig.add_subplot(3, 2, 2)
    axs3 = fig.add_subplot(3, 2, 3)
    axs4 = fig.add_subplot(3, 2, 4)
    axs5 = fig.add_subplot(3, 1, 3)

    x = np.arange(0, 11)
    axs1.set_xlim(0, 10)
    axs1.plot(x, y0, "r-")

    axs2.scatter(x1, y1, color='m')
    axs2.set_xlabel("Height (in)", fontsize = 'x-small')
    axs2.set_ylabel("Weight (lbs)", fontsize = 'x-small')
    axs2.set_title("Men's Height and Weight", fontsize = 'x-small')

    axs3.plot(x2, y2)
    axs3.set_title('Exponential Decay of C-14', fontsize = 'x-small')
    axs3.set_xlabel('Time (years)', fontsize = 'x-small')
    axs3.set_xlim([0, 28650])
    axs3.set_yscale('log')
    axs3.set_ylabel('Fraction Remaining', fontsize = 'x-small')

    axs4.set_title('Exponential Decay of Radioactive Elements', fontsize = 'x-small')
    axs4.set_xlabel('Time (years)', fontsize = 'x-small')
    axs4.set_ylim([0, 1])
    axs4.set_ylabel('Fraction Remaining', fontsize = 'x-small')
    axs4.plot(x3, y31, 'r--', x3, y32, 'g-')
    axs4.set_xlim([0, 20000])
    axs4.legend(['C-14', 'Ra-226'])

    axs5.set_title('Project A', fontsize = 'x-small')
    axs5.set_xlabel('Grades', fontsize = 'x-small')
    axs5.set_xlim([0, 100])
    axs5.set_ylabel('Number of Students', fontsize = 'x-small')
    axs5.set_ylim([0, 30])
    axs5.hist(student_grades, bins=np.arange(0, 101, 10), edgecolor='black')

    fig.suptitle('All in One')
    plt.tight_layout()
    plt.show()
