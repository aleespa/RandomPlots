import io

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt


def generate_plot(seed, bg_color, dark_mode):
    rng = np.random.default_rng(seed)

    dark_background_colormaps = [
        'Spectral', 'viridis', 'plasma', 'inferno', 'cividis',
        'YlOrRd', 'RdPu', 'spring', 'summer', 'autumn', 'winter',
        'cool', 'Wistia', 'hot', 'afmhot', 'copper', 'gist_heat',
        'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink'
    ]

    light_background_colormaps = [
        'gist_heat', 'binary', 'gist_yarg', 'gist_gray', 'gray',
        'bone', 'pink'
    ]
    if dark_mode:
        colormaps = dark_background_colormaps
    else:
        colormaps = light_background_colormaps
    colormap = rng.choice(colormaps, size=4)
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

    # Generate colormap
    colormaps = [mpl.colormaps[colormap[i % 4]] for i in range(sides)]
    for j in range(n):
        for s in range(sides):
            # Use colors based on the current segment
            color = colormaps[s](j / n)
            ax.plot(
                [x[j, s], x[n - j - 1, (s + 1) % sides]],
                [y[j, s], y[n - j - 1, (s + 1) % sides]],
                color=color,
                lw=1
            )

    ax.axis('off')

    buffer = io.BytesIO()
    plt.savefig(buffer, format='jpg', bbox_inches='tight', pad_inches=0)
    buffer.seek(0)

    return buffer


def create_image(seed=0, dark_mode=True, bg_color=(0, 0, 0)):
    buffer = generate_plot(seed, bg_color, dark_mode)
    plt.close()
    return buffer.getvalue()
