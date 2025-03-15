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

    l = 300
    n = 28
    cs = ColorSelector(mpl.colormaps[colormap], dark_mode)
    z_vals = np.linspace(-2 * np.pi, 2 * np.pi, n)
    w_vals = np.linspace(-2 * np.pi, 2 * np.pi, n)
    x_vals = np.linspace(0, 2 * np.pi, l)

    # Create grids for vectorized computations
    z_grid, w_grid = np.meshgrid(z_vals, w_vals, indexing='ij')
    a, b = rng.uniform(0.5, 1, 2) * rng.choice([-1, 1], 2)
    sum_zw = a * z_grid + b * w_grid
    a, b = rng.uniform(-1, 1, 2)
    radius = a * np.sin(sum_zw) + b * np.cos(sum_zw)

    # Precompute trigonometric functions for x values
    cos_x = 0.8 * np.cos(x_vals)
    sin_x = 0.8 * np.sin(x_vals)

    # Vectorized computation of all coordinates
    X = radius[..., np.newaxis] * cos_x + z_grid[..., np.newaxis]  # Shape (n, n, l)
    Y = radius[..., np.newaxis] * sin_x + w_grid[..., np.newaxis]  # Shape (n, n, l)

    # Reshape for plotting (flatten z/w dimensions, keep x as sequence)
    X_flat = X.reshape(-1, l).T  # Shape (l, nxn)
    Y_flat = Y.reshape(-1, l).T  # Shape (l, nxn)

    # Single plot call for all lines
    plt.plot(X_flat, Y_flat, alpha=0.9, color=cs.get_color(rng), lw=2.2)


    ax.axis('off')
    buffer = io.BytesIO()
    plt.savefig(buffer, format='jpg', bbox_inches='tight', pad_inches=0)
    buffer.seek(0)

    return buffer


def brownian_bridge(rng, n):
    t = np.linspace(0, 1, n)
    dW = rng.normal(size=n) * np.sqrt(1 / n)
    W = np.cumsum(dW)
    return W - t * W[-1]

def create_image(seed=0, dark_mode=True, bg_color=(0, 0, 0)):
    buffer = generate_plot(seed, bg_color, dark_mode)
    plt.close()
    return buffer.getvalue()


