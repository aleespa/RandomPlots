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

    n_sides = rng.choice([i for i in range(4, 10)] + [300])
    sides = np.linspace(0, 2 * np.pi, n_sides)
    n_pol = rng.integers(25, 80)
    for z in np.linspace(0, 5, n_pol):
        plt.plot(np.cos(sides + z) * z, np.sin(sides + z) * z, lw=3, color=cmap(z / 5))

    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)

    ax.axis('off')

    buffer = io.BytesIO()
    plt.savefig(buffer, format='jpg', bbox_inches='tight', pad_inches=0)
    buffer.seek(0)

    return buffer
