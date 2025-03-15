import io

import matplotlib as mpl
import numpy as np
from Colors.ColorSelector import ColorSelector
from matplotlib import pyplot as plt


def generate_plot(seed, bg_color=(0, 0, 0), dark_mode=True):
    rng = np.random.default_rng(seed)

    dark_background_colormaps = [
         'hsv',
        'Spectral', 'viridis', 'plasma', 'inferno', 'cividis',
        'YlOrRd', 'RdPu', 'spring', 'summer', 'autumn', 'winter',
        'cool', 'Wistia', 'hot', 'afmhot', 'copper', 'gist_heat',
        'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
        'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
        'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
        'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
        'hsv', 'rainbow'
    ]

    light_background_colormaps = [
        'gist_heat', 'binary', 'gist_yarg', 'gist_gray', 'gray',
        'bone', 'pink',
        'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
        'GnBu', 'PuBu', 'PuBuGn', 'BuGn', 'YlGn',
        'hsv', 'rainbow'
    ]
    if dark_mode:
        colormaps = dark_background_colormaps
    else:
        colormaps = light_background_colormaps

    colormap = rng.choice(colormaps)
    fig, ax = plt.subplots(figsize=(12, 12), dpi=200, tight_layout=True)
    fig.patch.set_facecolor(bg_color)
    n = rng.integers(10, 50)
    z = rng.uniform(-1, 1, (6, n))
    cs = ColorSelector(mpl.colormaps[colormap], dark_mode)
    alpha_value = min(0.65, max(20 / n, 0.2))
    for i in range(n):
        plt.fill_between(
            [z[0, i], z[1, i]],
            [z[2, i], z[3, i]],
            [z[4, i], z[5, i]],
            alpha=alpha_value,
            color=cs.get_color(rng),
            lw=0
        )
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.axis('off')
    buffer = io.BytesIO()
    plt.savefig(buffer, format='jpg', bbox_inches='tight', pad_inches=0)
    buffer.seek(0)

    return buffer


def create_image(seed=0, dark_mode=True, bg_color=(0, 0, 0)):
    buffer = generate_plot(seed, bg_color, dark_mode)
    plt.close()
    return buffer.getvalue()


def brownian_bridge(rng, n):
    t = np.linspace(0, 1, n)
    dW = rng.normal(size=n) * np.sqrt(1 / n)
    W = np.cumsum(dW)
    return W - t * W[-1]
