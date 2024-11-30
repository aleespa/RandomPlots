import base64
import io

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl


def generate_plot():
    colormaps = [
        'Spectral',
        'viridis',
        'plasma',
        'inferno',
        'cividis',
        'YlOrRd',
        'RdPu',
        'gist_heat',
        'afmhot',
        'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone',
        'pink', 'spring', 'summer', 'autumn', 'winter', 'cool',
        'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper'
    ]

    fig, ax = plt.subplots(figsize=(1, 1), dpi=50, tight_layout=True)
    fig.patch.set_facecolor('#ffff')

    n_sides = 7
    sides = np.linspace(0, 2 * np.pi, n_sides)

    for k in np.linspace(0, 30, 16):
        ax.plot(k * np.cos(sides) + np.cos(k + 1),
                k * np.sin(sides) + np.sin(k - 1),
                alpha=1,
                lw=np.random.uniform(0.1, 1),
                color='k')

    ax.set_xlim(-32, 32)
    ax.set_ylim(-32, 32)

    ax.axis('off')
    plt.savefig("polygon_tunnel.svg", format='svg', bbox_inches='tight', pad_inches=0, transparent=True)



if __name__ == "__main__":
    generate_plot()
