import numpy as np
from matplotlib import pyplot as plt, patches
from scipy.stats import alpha


def generate_fill():
    fig, ax = plt.subplots(figsize=(2, 2), dpi=50, tight_layout=True)
    fig.patch.set_facecolor('#000000')

    rect1 = patches.Rectangle((0, 0), 1, 1, linewidth=1, edgecolor='k', facecolor='k', alpha=0.5)
    rect2 = patches.Rectangle((1 / 3, 0), 1 / 3, 1, linewidth=1, edgecolor='k', facecolor='k', alpha=0.5)

    ax.add_patch(rect1)
    ax.add_patch(rect2)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.savefig("fill.svg", format='svg', bbox_inches='tight', pad_inches=0, transparent=True)


def generate_fit():
    fig, ax = plt.subplots(figsize=(2, 2), dpi=50, tight_layout=True)
    fig.patch.set_facecolor('#000000')

    rect1 = patches.Rectangle((1 / 3, 1 / 3), 1 / 3, 1 / 3, linewidth=1, edgecolor='k', facecolor='k', alpha=0.5)
    rect2 = patches.Rectangle((1 / 3, 0), 1 / 3, 1, linewidth=1, edgecolor='k', facecolor='k', alpha=0.5)

    ax.add_patch(rect1)
    ax.add_patch(rect2)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.axis('off')
    plt.savefig("fit.svg", format='svg', bbox_inches='tight', pad_inches=0, transparent=True)


if __name__ == "__main__":
    generate_fill()
    generate_fit()
