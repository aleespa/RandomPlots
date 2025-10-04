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

    l = 300
    n = 28
    z_vals = np.linspace(-2 * np.pi, 2 * np.pi, n)
    w_vals = np.linspace(-2 * np.pi, 2 * np.pi, n)
    x_vals = np.linspace(0, 2 * np.pi, l)

    z_grid, w_grid = np.meshgrid(z_vals, w_vals, indexing='ij')
    a, b = rng.uniform(0.5, 1, 2) * rng.choice([-1, 1], 2)
    sum_zw = a * z_grid + b * w_grid
    a, b = rng.uniform(-1, 1, 2)
    radius = a * np.sin(sum_zw) + b * np.cos(sum_zw)

    cos_x = 0.8 * np.cos(x_vals)
    sin_x = 0.8 * np.sin(x_vals)

    x = radius[..., np.newaxis] * cos_x + z_grid[..., np.newaxis]
    y = radius[..., np.newaxis] * sin_x + w_grid[..., np.newaxis]

    x_flat = x.reshape(-1, l).T
    y_flat = y.reshape(-1, l).T

    plt.plot(x_flat, y_flat, alpha=0.9, color=cmap(rng.uniform()), lw=2.2)

    ax.axis('off')
    buffer = io.BytesIO()
    plt.savefig(buffer, format='jpg', bbox_inches='tight', pad_inches=0)
    buffer.seek(0)

    return buffer
