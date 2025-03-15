import matplotlib.colors as mcolors
import numpy as np
from matplotlib import pyplot as plt

colors = ["#ffffff", "#002e6b", '#000000']
cmap = mcolors.LinearSegmentedColormap.from_list("mycmap", colors, N=256)


def generate_svg():
    fig, ax = plt.subplots(figsize=(2, 2), dpi=50, tight_layout=True)

    n_sides = 4
    sides = np.linspace(0, 2 * np.pi, n_sides + 1)
    for r in np.linspace(0, np.pi, n_sides + 2)[1:]:
        plt.plot(np.cos(sides + r) * r / np.pi,
                 np.sin(sides + r) * r / np.pi,
                 lw=4.2 + (r/np.pi),
                 color=cmap(r / np.pi),
                 solid_capstyle='round')

    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.8, 1.8)
    ax.axis('off')
    plt.savefig("main_icon.svg", format='svg', bbox_inches='tight', pad_inches=0, transparent=True)


def generate_png():
    fig, ax = plt.subplots(figsize=(5, 5), dpi=109, tight_layout=True)
    fig.patch.set_facecolor('#f4f0e7')
    n_sides = 4
    sides = np.linspace(0, 2 * np.pi, n_sides + 1)
    for r in np.linspace(0, np.pi, n_sides + 2)[1:]:
        plt.plot(np.cos(sides + r) * r / np.pi,
                 np.sin(sides + r) * r / np.pi,
                 lw=2.6 * (4.2 + (r / np.pi)),
                 color=cmap(r / np.pi),
                 solid_capstyle='round')

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.axis('off')
    plt.savefig("main_icon.png", format='png', bbox_inches='tight', pad_inches=0)


if __name__ == "__main__":
    generate_svg()
    generate_png()
