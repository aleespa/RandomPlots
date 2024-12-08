import base64
import io

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import gc


def generate_plot(seed, dark_mode, bg_color):
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
    colormap = rng.choice(colormaps)
    fig, ax = plt.subplots(figsize=(12, 12), dpi=200, tight_layout=True)
    fig.patch.set_facecolor(bg_color)
    variance = np.random.exponential(1)
    n = 150
    for z in np.linspace(1, 20, 100):
        theta = np.linspace(0, 2 * np.pi, n)
        c, s = np.cos(theta), np.sin(theta)
        r0 = np.random.uniform(1 * z, (1+variance) * z)
        r = np.array([r0] + list(np.random.uniform(1 * z, (1+variance) * z, n - 2)) + [r0])

        ax.plot(c*r, r*s,
                lw=1.1,
                color=mpl.colormaps[colormap](z / 20))

    ax.axis('off')

    buffer = io.BytesIO()
    plt.savefig(buffer, format='jpg', bbox_inches='tight', pad_inches=0)
    buffer.seek(0)

    return buffer


def create_image(seed=0, dark_mode=True, bg_color=(0, 0, 0)) -> str:
    buffer = generate_plot(seed, dark_mode, bg_color)
    image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    gc.collect()
    return image_data
