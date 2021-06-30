import math

import numpy as np
from matplotlib import pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def plot(px, py):
    plt.plot(px, py)
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data',0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data',0))
    plt.show()


def main():
    # Init
    x = []
    dx = -20
    while dx <= 20:
        x.append(dx)
        dx += 0.1

    # Use sigmoid() function
    px = [xv for xv in x]
    print(px)
    py = [sigmoid(xv) for xv in x]
    print(py)

    # Plot
    plot(px, py)


if __name__ == "__main__":
    main()
