import sys

import numpy as np
from matplotlib import pyplot as plt


def calculate_random_points(sample_size):
    # Sample 2n random angles from 0 to 2*pi (2 for each pair)
    thetas = np.random.uniform(0, 2 * np.pi, size=2 * sample_size)

    r = np.random.choice(np.linspace(0, 1, 200)[1:], size=2 * sample_size )

    # Compute complex numbers
    zs = np.exp(1j * thetas) * r

    # Reshape to get n pairs
    pairs = zs.reshape(sample_size, 2)

    return pairs


def generate_plot(x, y, filename):
    fig, _ = plt.subplots(figsize=(12, 12), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1], facecolor='#f4f0e7')
    ax.scatter(x, y, s=30, color='k', lw=0, alpha=0.9)
    ax.set_xlim(-0.25, 0.25)
    ax.set_ylim(-0.25, 0.25)
    fig.savefig(f'outputs/{filename}.png', facecolor='k')
    plt.close()


def generate():
    filename = sys.argv[1]
    sample_size = 20000
    Z = calculate_random_points(sample_size)
    x = Z.real
    y = Z.imag
    generate_plot(x, y, filename)
