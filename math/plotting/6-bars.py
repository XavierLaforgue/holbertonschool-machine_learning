#!/usr/bin/env python3
"""Define function bars."""
import matplotlib.pyplot as plt
import numpy as np


def bars():
    """Plot a stacked bar graph."""
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))
    people = ("Farrah", "Fred", "Felicia")
    fruits = ("apples", "bananas", "oranges", "peaches")
    colors = ("red", "yellow", "#ff8000", "#ffe5b4")
    new_bottom = np.zeros(3)
    for i in range(4):
        plt.bar(people, fruit[i, :],
                width=.5, bottom=new_bottom,
                color=colors[i], label=fruits[i])
        new_bottom += fruit[i, :]
    plt.legend(loc="upper right")
    plt.ylabel('Quantity of Fruit')
    plt.ylim(0, 80)
    plt.yticks(np.arange(0, 81, 10))
    plt.title('Number of Fruit per Person')
    # plt.subplots_adjust(left=0.05, right=.95,
    #                 bottom=0.05, top=.95)
    plt.show()
