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
    n = 750
    x = np.linspace(0, 2 * np.pi, n)
    n_waves = rng.integers(12, 22)
    lws = np.random.uniform(1, 4, size=n_waves)
    cs = ColorSelector(mpl.colormaps[colormap], dark_mode)
    for k in range(n_waves):
        bb = brownian_bridge(rng, n)
        i = np.arange(0, n)
        y = np.sin(bb) * np.cos(i) + k * 1.6
        ax.plot(x, y,
                color=cs.get_color(rng),
                lw=lws[k],
                alpha=1
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
    dW = rng.normal(size=n) * np.sqrt(1/n)
    W = np.cumsum(dW)
    return W - t * W[-1]

