import base64
import io

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl


def generate_plot(seed, dark_mode, bg_color):
    rng = np.random.default_rng(seed)

    dark_background_colormaps = [
        'Spectral', 'viridis', 'plasma', 'inferno', 'cividis',
        'YlOrRd', 'RdPu', 'spring', 'summer', 'autumn', 'winter',
        'cool', 'Wistia', 'hot', 'afmhot', 'copper','gist_heat',
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

    fig, ax = plt.subplots(figsize=(12, 12), dpi=200, tight_layout=True)
    fig.patch.set_facecolor(bg_color)

    n_sides = rng.choice([i for i in range(5, 13)] + [300])
    r = np.linspace(0, 30, 90)
    sides = np.linspace(0, 2 * np.pi, n_sides)

    colormap = rng.choice(colormaps)
    norm, scale = rng.integers(10, 25), rng.integers(30, 60)
    for k in np.linspace(0, 30, 100):
        ax.plot(k * np.cos(sides) + np.cos(k + 1),
                k * np.sin(sides) + np.sin(k - 1),
                alpha=0.7,
                lw=rng.uniform(0.5, 4),
                color=mpl.colormaps[colormap]((k + norm) / scale))

    ax.set_xlim(-32, 32)
    ax.set_ylim(-32, 32)

    ax.axis('off')

    buffer = io.BytesIO()
    plt.savefig(buffer, format='jpg', bbox_inches='tight', pad_inches=0)
    buffer.seek(0)

    return buffer


def create_image(seed=0, dark_mode=True, bg_color=(0, 0, 0)) -> str:
    buffer = generate_plot(seed, dark_mode, bg_color)
    image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    return image_data
