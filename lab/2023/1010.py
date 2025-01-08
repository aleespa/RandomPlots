import sys

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


def calculate_matrix(t: np.array):
    return np.array(
        [
            [1j, -1, 1, 1, 1j],
            [-1, 1, 0, 1, 1],
            [t[1], 0, -1j, 0, 1j],
            [1j, t[0], 1j, 1j, 1j],
            [1j, 2, -1, -1, 1j],
        ]
    )


def calculate_eigenvalues(x: np.array):
    return np.linalg.eigvals(x)


def generate_plot(x, y, filename):
    fig, _ = plt.subplots(figsize=(12, 12), dpi=200)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0), facecolor='#f4f0e7')
    ax.scatter(x, y, s=1, color='k', lw=0, alpha=0.9)
    ax.set_xlim(-2.5, 1.5)
    ax.set_ylim(-1.5, 2.5)
    fig.savefig(f'outputs/{filename}.png', facecolor='k')
    plt.close()


def generate():
    filename = sys.argv[1]
    sample_size = 50000
    sample = vectorized_sample_complex_pairs(sample_size)
    Z = np.array([calculate_eigenvalues(calculate_matrix(t)) for t in sample]).ravel()
    x = Z.real
    y = Z.imag
    generate_plot(x, y, filename)
    print(x.shape)
