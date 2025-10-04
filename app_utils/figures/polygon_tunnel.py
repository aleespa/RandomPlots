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

    n_sides = rng.choice([i for i in range(5, 13)] + [300])
    r = np.linspace(0, 30, 90)
    sides = np.linspace(0, 2 * np.pi, n_sides)

    norm, scale = rng.integers(10, 25), rng.integers(30, 120)
    for k in np.linspace(0, 30, 100):
        ax.plot(
            k * np.cos(sides) + np.cos(k + 1),
            k * np.sin(sides) + np.sin(k - 1),
            alpha=0.7,
            lw=rng.uniform(0.5, 4),
            color=cmap((k + norm) / scale),
        )

    ax.set_xlim(-32, 32)
    ax.set_ylim(-32, 32)

    ax.axis('off')

    buffer = io.BytesIO()
    plt.savefig(buffer, format='jpg', bbox_inches='tight', pad_inches=0)
    buffer.seek(0)

    return buffer
