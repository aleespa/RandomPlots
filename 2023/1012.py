import os
import sys
import time
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt


def vectorized_sample_complex_pairs(sample_size: int):
    # Sample 2n random angles from 0 to 2*pi (2 for each pair)
    thetas = np.random.uniform(0, 2 * np.pi, size=2 * sample_size)

    # Compute complex numbers
    zs = np.exp(1j * thetas)

    # Reshape to get n pairs
    pairs = zs.reshape(sample_size, 2)

    return pairs


def calculate_matrix(t):
    return np.array([[-1j, 0, -1j, 0.5, -1j],
                     [-1j, 1, -1j, 0, 0],
                     [0,  t[1], -1j, 0.5, 1],
                     [1, -1j, 1j, 0.5, 1j],
                     [0, 1j, t[0], 0, 1]])
    # return np.array([[t[0], 1j],
    #                  [-0.5, t[1]]])


def calculate_eigenvalues(x: np.array):
    return np.linalg.eigvals(x)


def generate_plot(x, y, directory):
    current_time = datetime.now()
    time_string = current_time.strftime('%Y-%m-%d_%H-%M-%S')
    fig, _ = plt.subplots(figsize=(12, 12), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1], facecolor='#f4f0e7')
    ax.scatter(x, y, color='k', s=5, lw=0)
    if not os.path.exists(f"outputs/{directory}"):
        os.makedirs(f"outputs/{directory}")
    fig.savefig(f'outputs/{directory}/{time_string}.png', facecolor='k')
    plt.close()


def generate():
    directory = sys.argv[1]
    sample_size = 50000
    sample = vectorized_sample_complex_pairs(sample_size)
    for r in np.linspace(1, 3, 100):
        Z = np.array([calculate_eigenvalues(calculate_matrix(t)) for t in sample * r]).ravel()
        generate_plot(Z.real, Z.imag, directory)
