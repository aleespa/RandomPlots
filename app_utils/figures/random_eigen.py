import io

import numpy as np
from matplotlib import pyplot as plt


def vectorized_sample_complex_pairs(rng, sample_size: int):
    thetas = rng.uniform(0, 2 * np.pi, size=(sample_size, 3))
    return np.exp(1j * thetas)


def calculate_matrix(m,
                     t: np.array,
                     r,
                     indexes):
    m[indexes[0]] = t[0]
    m[indexes[1]] = t[1]
    m[indexes[2]] = t[2]
    return m


def calculate_eigenvalues(x: np.array):
    return np.linalg.eigvals(x)


def generate_plot(z, dark_mode=False, bg_color=(0, 0, 0)):
    # Create a figure with adjusted layout
    fig, ax = plt.subplots(figsize=(12, 12), dpi=100, tight_layout=True)
    fig.patch.set_facecolor(bg_color)
    # Create scatter plot without axes
    ax.scatter(z.real, z.imag, s=2,
               color='w' if dark_mode else "k",
               lw=0, alpha=0.9)
    x_center, y_center = (np.mean(z.real), np.mean(z.imag))
    ax.set_xlim(x_center - 8, x_center + 8)
    ax.set_ylim(y_center - 8, y_center + 8)

    # Remove axes lines and labels
    ax.axis('off')

    # Save the figure to a buffer without extra white space
    buffer = io.BytesIO()
    plt.savefig(buffer, format='jpg', bbox_inches='tight', pad_inches=0)
    buffer.seek(0)

    return buffer


def create_image(seed=0, dark_mode=True, bg_color=(0, 0, 0)):
    rng = np.random.default_rng(seed)
    sample_size = 10000
    sample = vectorized_sample_complex_pairs(rng, sample_size)
    r = rng.uniform(-1, 1, 3) + rng.uniform(-1, 1, 3) * 1j
    n = rng.integers(3, 12)
    m = rng.integers(-1, 2, (n, n)) + 1j * rng.integers(-1, 2, (n, n))
    indexes = rng.integers(0, n, (3, 2))

    modified_matrices = np.array([calculate_matrix(m.copy(), t, r, indexes) for t in sample])
    Z = np.linalg.eigvals(modified_matrices).ravel()
    buffer = generate_plot(Z, dark_mode, bg_color)
    plt.close()
    return buffer.getvalue()
