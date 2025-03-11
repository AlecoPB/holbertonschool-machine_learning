#!/usr/bin/env python3
"""
Bar graph
"""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """
    Source code to plot a stacked bar graph
    """
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))

    names = ['Farrah', 'Fred', 'Felicia']
    colors = ['red', 'yellow', '#ff8000', '#ffe5b4']
    fruit_names = ['apples', 'bananas', 'oranges', 'peaches']

    for i in range(fruit.shape[0]):
        plt.bar(names,
                fruit[i],
                bottom=np.sum(fruit[:i], axis=0),
                color=colors[i],
                label=fruit_names[i],
                width=0.5)

    plt.title('Number of Fruit per Person')
    plt.ylabel('Quantity of Fruit')
    plt.ylim([0, 80])
    plt.yticks(np.arange(0, 81, 10))

    plt.legend()
    plt.show()
