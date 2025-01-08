import os
import sys
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt


def vectorized_sample_complex_pairs(num_points=100):
    """
    Generate a meshgrid of points in the interval (a, b) x (c, d).

    Parameters:
    - a, b, c, d: The boundaries of the interval.
    - num_points: The number of points along each axis.

    Returns:
    - X, Y: Meshgrid of x and y coordinates.
    """
    # x = np.linspace(a, b, num_points)
    # y = np.linspace(c, d, num_points)
    r = np.random.uniform(-20, 20, size=(num_points, 2))
    return r


def calculate_matrix(t):
    return np.array(
        [
            [-1j, 0, -1j, 0.5, -1j],
            [-1j, 1, -1j, 0, 0],
            [0, 0, -1j, 0.5, 1],
            [1, -1j, 1j, 0.5, 1j],
            [0, 1j, t[0], 0, 1],
        ]
    )
    # return np.array([[t[0], 1j],
    #                  [-0.5, t[1]]])


def calculate_eigenvalues(x: np.array):
    return np.linalg.eigvals(x)


def generate_plot(x, y, directory):
    current_time = datetime.now()
    time_string = current_time.strftime('%Y-%m-%d_%H-%M-%S')
    fig, _ = plt.subplots(figsize=(12, 12), dpi=400)
    ax = fig.add_axes([0, 0, 1, 1], facecolor='#f4f0e7')
    ax.scatter(x, y, s=1, color='k', lw=0, alpha=0.9)
    if not os.path.exists(f"outputs/{directory}"):
        os.makedirs(f"outputs/{directory}")
    fig.savefig(f'outputs/{directory}/{time_string}.png', facecolor='k')
    plt.close()


def generate():
    directory = sys.argv[1]
    sample_size = 500000
    sample = vectorized_sample_complex_pairs(sample_size)
    Z = np.array([calculate_eigenvalues(calculate_matrix(t)) for t in sample]).ravel()
    x = Z.real
    y = Z.imag
    generate_plot(x, y, directory)
    print(x.shape)
