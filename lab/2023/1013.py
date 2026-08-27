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


def calculate_matrix_v1(t):
    return np.array([[t[0], 1j], [0.5, t[1]]])


def calculate_matrix_v2(t):
    return np.array([[t[0], 1j], [-0.5, t[1]]])


def calculate_eigenvalues(x: np.array):
    return np.linalg.eigvals(x)


def generate_plot(x, y, i, color, settings: ImageProcessingSettings):
    fig, _ = plt.subplots(figsize=(9, 16), dpi=100)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0), facecolor='#f4f0e7')
    ax.scatter(x[0], y[1], s=5, lw=0, color=color[0])
    ax.scatter(x[1], y[0], s=5, lw=0, color=color[1])
    y1, y2 = -2, 2
    x1, x2 = -2, 2
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
    for i, r in enumerate(np.linspace(0, 4.8, 360)):
        Z1 = np.array(
            [calculate_eigenvalues(calculate_matrix_v1(t)) for t in sample * r]
        ).ravel()
        Z2 = np.array(
            [calculate_eigenvalues(calculate_matrix_v2(t)) for t in sample * r]
        ).ravel()
        generate_plot(
            [Z1.real, Z2.real], [Z1.imag, Z2.imag], i, ['#cc0000', '#000033'], settings
        )
    settings.save_video(30)


if __name__ == '__main__':
    generate()
