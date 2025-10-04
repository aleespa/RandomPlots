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
    n = rng.integers(10, 50)
    z = rng.uniform(-1, 1, (6, n))
    alpha_value = min(0.65, max(20 / n, 0.2))
    for i in range(n):
        plt.fill_between(
            [z[0, i], z[1, i]],
            [z[2, i], z[3, i]],
            [z[4, i], z[5, i]],
            alpha=alpha_value,
            color=cmap(rng.uniform()),
            lw=0,
        )
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.axis('off')
    buffer = io.BytesIO()
    plt.savefig(buffer, format='jpg', bbox_inches='tight', pad_inches=0)
    buffer.seek(0)

    return buffer
