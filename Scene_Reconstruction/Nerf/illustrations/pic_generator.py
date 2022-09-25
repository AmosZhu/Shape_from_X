"""
Author: dizhong zhu
Date: 25/09/2022
"""

import numpy as np
import matplotlib.pyplot as plt


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def density_function():
    pass


if __name__ == '__main__':
    x_values = np.linspace(0, 80, 120)
    y_values_1 = gaussian(x_values, mu=20, sig=2)
    # plt.plot(x_values, y_values)

    y_values_2 = gaussian(x_values, mu=50, sig=6) * 0.3
    plt.plot(x_values, y_values_1 + y_values_2)

    plt.show()
