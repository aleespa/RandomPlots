import base64
import io

import matplotlib as mpl
import matplotlib.colors as mcolors
import numpy as np
from Colors.ColorSelector import ColorSelector
from matplotlib import colormaps as cmaps
from matplotlib import pyplot as plt

original1 = mcolors.LinearSegmentedColormap.from_list(
    'original1',
    ['#96ceb4', '#ffeead', '#ff6f69', '#ffcc5c', '#88d8b0','#f7f4a3', '#7fccec', '#6a81d9', '#a479c9', '#dfdfdf']
)
cmaps.register(cmap=original1)


def generate_plot(seed, bg_color=(0, 0, 0), dark_mode=True):
    rng = np.random.default_rng(seed)

    dark_background_colormaps = [
        'original1',
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
    n = rng.integers(20, 65)
    x = rng.uniform(0, 1, size=(2, n))
    ax.scatter(x[0], x[1], s=120,
               color='#ffffff' if dark_mode else '#000000',
               zorder=2)
    distances = np.sqrt((x[0][:, np.newaxis] - x[0]) ** 2 + (x[1][:, np.newaxis] - x[1]) ** 2)

    # Plot lines between points that are within a radius of 0.2
    radius = rng.uniform(0.15, 0.3)
    cs = ColorSelector(mpl.colormaps[colormap], dark_mode)
    for i in range(n):
        for j in range(i + 1, n):
            if distances[i, j] < radius:
                ax.plot([x[0][i], x[0][j]], [x[1][i], x[1][j]],
                        color=cs.get_color(rng),
                        lw=1.2,
                        zorder=1
                        )

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
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
    dW = rng.normal(size=n) * np.sqrt(1/n)
    W = np.cumsum(dW)
    return W - t * W[-1]

