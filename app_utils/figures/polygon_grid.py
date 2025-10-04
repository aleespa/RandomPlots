import io

import numpy as np
from matplotlib import pyplot as plt


def create_image(seed=0, bg_color=(0, 0, 0), cmap=None):
    buffer = generate_plot(seed, bg_color, cmap)
    plt.close()
    return buffer.getvalue()


def generate_plot(seed, bg_color=(0, 0, 0), cmap=None):
    rng = np.random.default_rng(seed)

    n = 30
    sides = rng.integers(8, 15)
    min_radius = rng.uniform(0.2, 0.7)
    r = np.linspace(min_radius, 1, n)
    angles = np.linspace(0, 2 * np.pi, sides, endpoint=False)

    # Compute points using vectorized operations
    x = np.outer(r, np.cos(angles))  # Shape: (n, sides)
    y = np.outer(r, np.sin(angles))  # Shape: (n, sides)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 12), dpi=200, tight_layout=True)
    fig.patch.set_facecolor(bg_color)

    for j in range(n):
        for s in range(sides):
            color = cmap(j / n)
            ax.plot(
                [x[j, s], x[n - j - 1, (s + 1) % sides]],
                [y[j, s], y[n - j - 1, (s + 1) % sides]],
                color=color,
                lw=1,
            )

    ax.axis('off')

    buffer = io.BytesIO()
    plt.savefig(buffer, format='jpg', bbox_inches='tight', pad_inches=0)
    buffer.seek(0)

    return buffer
