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
    n = rng.integers(20, 65)
    x = rng.uniform(0, 1, size=(2, n))
    ax.scatter(
        x[0],
        x[1],
        s=120,
        color='#ffffff' if is_dark_color(bg_color) else '#000000',
        zorder=2,
    )
    distances = np.sqrt(
        (x[0][:, np.newaxis] - x[0]) ** 2 + (x[1][:, np.newaxis] - x[1]) ** 2
    )

    radius = rng.uniform(0.15, 0.3)
    for i in range(n):
        for j in range(i + 1, n):
            if distances[i, j] < radius:
                ax.plot(
                    [x[0][i], x[0][j]],
                    [x[1][i], x[1][j]],
                    color=cmap(rng.uniform()),
                    lw=1.2,
                    zorder=1,
                )

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.axis('off')
    buffer = io.BytesIO()
    plt.savefig(buffer, format='jpg', bbox_inches='tight', pad_inches=0)
    buffer.seek(0)

    return buffer


def is_dark_color(hex_color: str, threshold: float = 0.5) -> bool:
    """
    Determine if a hex color is dark.

    Args:
        hex_color: String in format "#RRGGBB" or "RRGGBB".
        threshold: Luminance threshold (0 = darkest, 1 = brightest). Default 0.5.

    Returns:
        True if the color is dark, False otherwise.
    """
    # Remove hash if present
    hex_color = hex_color.lstrip("#")

    # Convert to RGB integers
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    # Compute relative luminance
    luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255

    # Return True if dark
    return luminance < threshold