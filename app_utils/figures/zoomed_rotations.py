import base64
import io

import matplotlib as mpl
import numpy as np
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
        'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'
    ]

    light_background_colormaps = [
        'gist_heat', 'binary', 'gist_yarg', 'gist_gray', 'gray',
        'bone', 'pink',
        'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
        'GnBu', 'PuBu', 'PuBuGn', 'BuGn', 'YlGn'
    ]
    if dark_mode:
        colormaps = dark_background_colormaps
    else:
        colormaps = light_background_colormaps

    colormap = rng.choice(colormaps)
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
                color=mpl.colormaps[colormap](t / np.pi),
                alpha=0.8,
                lw=(r + 0.1),
            )

    ax.axis('off')
    buffer = io.BytesIO()
    plt.savefig(buffer, format='jpg', bbox_inches='tight', pad_inches=0)
    buffer.seek(0)

    return buffer


def create_image(seed=0, dark_mode=True, bg_color=(0, 0, 0)):
    buffer = generate_plot(seed, bg_color, dark_mode)
    image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    return image_data


def brownian_bridge(rng, n):
    t = np.linspace(0, 1, n)
    dW = rng.normal(size=n) * np.sqrt(1 / n)
    W = np.cumsum(dW)
    return W - t * W[-1]
