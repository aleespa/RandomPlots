import base64
import io

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl


def generate_plot():
    fig, ax = plt.subplots(figsize=(2, 2), dpi=50, tight_layout=True)
    fig.patch.set_facecolor('#ffffff')

    n_sides = 5
    sides = np.linspace(0, 2 * np.pi, n_sides)
    for z in np.linspace(0, 5, 10):
        plt.plot(np.cos(sides + z) * z,
                 np.sin(sides + z) * z,
                 lw=3,
                 color='k')

    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)

    ax.axis('off')

    ax.axis('off')
    plt.savefig("polygon_feedback.svg", format='svg', bbox_inches='tight', pad_inches=0, transparent=True)

if __name__ == '__main__':
    generate_plot()
