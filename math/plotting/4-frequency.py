#!/usr/bin/env python3
"""
This is some documentation
"""
import numpy as np
import matplotlib.pyplot as plt

    
def frequency():
    """
    Histogram plotting
    """
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))

    plt.title('Project A')
    plt.xlabel('Grades')
    plt.xlim(0, 100)
    plt.ylabel('Number of Students')
    plt.ylim(0, 30)
    plt.hist(student_grades, bins=np.arange(0, 120, 10),
             edgecolor='black')
