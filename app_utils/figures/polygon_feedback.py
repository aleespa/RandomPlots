import io

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt


def generate_plot(seed, bg_color, dark_mode):
    rng = np.random.default_rng(seed)

    dark_background_colormaps = [
        'Spectral', 'viridis', 'plasma', 'inferno', 'cividis',
        'YlOrRd', 'RdPu', 'spring', 'summer', 'autumn', 'winter',
        'cool', 'Wistia', 'hot', 'afmhot', 'copper','gist_heat',
        'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink'
    ]

    light_background_colormaps = [
        'gnuplot', 'binary', 'gist_heat', 'rainbow'
    ]

    if dark_mode:
        colormaps = dark_background_colormaps
    else:
        colormaps = light_background_colormaps

    fig, ax = plt.subplots(figsize=(12, 12), dpi=200, tight_layout=True)
    fig.patch.set_facecolor(bg_color)

    n_sides = rng.choice([i for i in range(4, 10)] + [300])
    sides = np.linspace(0, 2 * np.pi, n_sides)
    colormap = rng.choice(colormaps)
    n_pol = rng.integers(25, 80)
    for z in np.linspace(0, 5, n_pol):
        plt.plot(np.cos(sides + z) * z,
                 np.sin(sides + z) * z,
                 lw=3,
                 color=mpl.colormaps[colormap](z / 5))

    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)

    ax.axis('off')

    buffer = io.BytesIO()
    plt.savefig(buffer, format='jpg', bbox_inches='tight', pad_inches=0)
    buffer.seek(0)

    return buffer


def create_image(seed=0, dark_mode=True, bg_color=(0, 0, 0)):
    buffer = generate_plot(seed, bg_color, dark_mode)
    plt.close()
    return buffer.getvalue()
