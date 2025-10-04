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
    n = 750
    x = np.linspace(0, 2 * np.pi, n)
    n_waves = rng.integers(12, 22)
    lws = np.random.uniform(1, 4, size=n_waves)
    for k in range(n_waves):
        bb = brownian_bridge(rng, n)
        i = np.arange(0, n)
        y = np.sin(bb) * np.cos(i) + k * 1.6
        ax.plot(x, y, color=cmap(rng.uniform()), lw=lws[k], alpha=1)

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
