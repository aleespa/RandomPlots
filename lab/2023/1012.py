import numpy as np
from matplotlib import pyplot as plt

from common.image_processing import ImageProcessingSettings


def vectorized_sample_complex_pairs(sample_size: int, rng: np.random.Generator):
    # Sample 2n random angles from 0 to 2*pi (2 for each pair)
    thetas = rng.uniform(0, 2 * np.pi, size=2 * sample_size)

    # Compute complex numbers
    zs = np.exp(1j * thetas)

    # Reshape to get n pairs
    pairs = zs.reshape(sample_size, 2)

    return pairs


def calculate_matrix(t):
    return np.array(
        [
            [-1j, 0, -1j, 0.5, -1j],
            [-1j, 1, -1j, 0, 0],
            [0, t[1], -1j, 0.5, 1],
            [1, -1j, 1j, 0.5, 1j],
            [0, 1j, t[0], 0, 1],
        ]
    )
    # return np.array([[t[0], 1j],
    #                  [-0.5, t[1]]])


def calculate_eigenvalues(x: np.array):
    return np.linalg.eigvals(x)


def generate_plot(x, y, i, settings: ImageProcessingSettings):
    fig, _ = plt.subplots(figsize=(9, 16), dpi=100)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0), facecolor='#f4f0e7')
    ax.scatter(x, y, color='k', s=5, lw=0)
    y1, y2 = -1, 1
    x1, x2 = -1, 1
    w = x2 - x1
    h = y2 - y1
    z = (16 / 18) * w - (1 / 2) * h
    ax.set_xlim(x1, x2)
    ax.set_ylim(y1 - z, y2 + z)
    settings.save_numbered_frame(i, 'k')
    del x, y


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)
    sample_size = 5000
    sample = vectorized_sample_complex_pairs(sample_size, settings.rng)
    for i, r in enumerate(np.linspace(0, 10, 600)):
        Z = np.array(
            [calculate_eigenvalues(calculate_matrix(t)) for t in sample * r]
        ).ravel()
        generate_plot(Z.real, Z.imag, i, settings)
    settings.save_video(20)


if __name__ == '__main__':
    generate()
