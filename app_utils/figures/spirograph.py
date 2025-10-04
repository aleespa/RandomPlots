import io

import numpy as np
from matplotlib import pyplot as plt


def create_image(seed=0, bg_color=(0, 0, 0), cmap=None):
    buffer = generate_plot(seed, bg_color, cmap)
    plt.close()
    return buffer.getvalue()


def generate_plot(seed, bg_color=(0, 0, 0), cmap=None):
    rng = np.random.default_rng(seed)
    # Generate data
    t = np.linspace(0, 2 * np.pi, 5000)
    fig, ax = plt.subplots(figsize=(12, 12), dpi=200, tight_layout=True)
    fig.patch.set_facecolor(bg_color)
    n_spirographs = rng.integers(4, 6)
    for _ in range(n_spirographs):
        a, b, c, d = generate_k_l(rng)
        plot_spiro(t, a, b, c, d, ax, cmap(rng.uniform()))

    # Ensure the center is always at (0, 0)
    ax.set_aspect('equal')  # Ensure equal scaling for x and y axes
    ax.spines['left'].set_position('zero')  # Move y-axis to x=0
    ax.spines['bottom'].set_position('zero')  # Move x-axis to y=0
    ax.spines['top'].set_visible(False)  # Hide top spine
    ax.spines['right'].set_visible(False)  # Hide right spine

    # Automatically calculate limits and make them symmetrical
    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()
    max_limit = max(abs(x_limits[0]), x_limits[1], abs(y_limits[0]), y_limits[1])
    ax.set_xlim(-max_limit, max_limit)
    ax.set_ylim(-max_limit, max_limit)

    # Remove axis ticks and labels
    ax.axis('off')

    # Save the figure to a buffer without extra white space
    buffer = io.BytesIO()
    plt.savefig(buffer, format='jpg', bbox_inches='tight', pad_inches=0)
    buffer.seek(0)

    return buffer


def spiro(t: np.array, k: float = 0.5, l: float = 0.5):
    return (1 - k) * np.exp(1j * t) + k * l * np.exp(-1j * t * (1 - k) / k)


def plot_spiro(t, a, b, c, d, ax, color):
    scaled_t = t * a
    s = spiro(scaled_t, a / b, c / d)
    ax.plot(s.real, s.imag, lw=3.5, alpha=0.9, color=color)


def generate_k_l(rng):
    while True:
        a, b, c, d = sorted(rng.integers(3, 35, 4))
        k, l = a / b, c / d
        if k != 1 and l != 1 and k != l:  # Ensure k and l are not 1
            return a, b, c, d
