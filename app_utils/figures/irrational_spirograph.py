import io

import numpy as np
from matplotlib import pyplot as plt


def create_image(seed=0, bg_color=(0, 0, 0), cmap=None):
    buffer = generate_plot(seed, bg_color, cmap)
    plt.close()
    return buffer.getvalue()


def generate_plot(seed, bg_color=(0, 0, 0), cmap=None):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi, 10000)
    fig, ax = plt.subplots(figsize=(12, 12), dpi=200, tight_layout=True)
    fig.patch.set_facecolor(bg_color)

    for _ in range(4):
        k, l = rng.uniform(0.01, 0.99, 2)
        plot_spiro(t, k, l, ax, cmap(rng.uniform()))

    ax.axis('off')
    buffer = io.BytesIO()
    plt.savefig(buffer, format='jpg', bbox_inches='tight', pad_inches=0)
    buffer.seek(0)

    return buffer


def spiro(t: np.ndarray, k: float = 0.5, l: float = 0.5):
    return (1 - k) * np.exp(1j * t) + k * l * np.exp(-1j * t * (1 - k) / k)


def plot_spiro(t, k, l, ax, color):
    s = spiro(100 * t, k, l)
    ax.plot(s.real, s.imag, lw=1, alpha=0.9, color=color)
