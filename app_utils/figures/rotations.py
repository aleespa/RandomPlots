import io

import numpy as np
from matplotlib import pyplot as plt


def create_image(seed=0, bg_color=(0, 0, 0), cmap=None):
    buffer = generate_plot(seed, bg_color, cmap)
    plt.close()
    return buffer.getvalue()


def generate_plot(seed, bg_color=(0, 0, 0), cmap=None):
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(12, 12), dpi=200, tight_layout=True)
    fig.patch.set_facecolor(bg_color)
    n = rng.integers(4, 12)
    i = rng.integers(4, 8)
    k = rng.integers(50, 70)
    theta = np.linspace(0, 2 * np.pi, i)
    for r in np.linspace(0, 1, n):
        for t in np.linspace(0, np.pi, k):
            plt.plot(
                r * np.cos(theta + t),
                r * np.sin(theta + t),
                color=cmap(t / np.pi),
                alpha=0.8,
                lw=(r + 0.1),
            )

    ax.axis('off')
    buffer = io.BytesIO()
    plt.savefig(buffer, format='jpg', bbox_inches='tight', pad_inches=0)
    buffer.seek(0)

    return buffer
