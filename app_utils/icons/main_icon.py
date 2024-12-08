import numpy as np
from matplotlib import pyplot as plt


def generate_svg():
    fig, ax = plt.subplots(figsize=(2, 2), dpi=50, tight_layout=True)
    fig.patch.set_facecolor('#000000')

    n_sides = 5
    sides = np.linspace(0, 2 * np.pi, n_sides)
    for z in np.linspace(0, 5, 6):
        plt.plot(np.cos(sides + z) * z,
                 np.sin(sides + z) * z,
                 lw=4,
                 color=plt.cm.gist_yarg_r(z / 25))

    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)
    ax.axis('off')
    plt.savefig("main_icon.svg", format='svg', bbox_inches='tight', pad_inches=0, transparent=True)


def generate_png():
    fig, ax = plt.subplots(figsize=(5, 5), dpi=109, tight_layout=True)
    fig.patch.set_facecolor('w')

    n_sides = 5
    sides = np.linspace(0, 2 * np.pi, n_sides)
    for z in np.linspace(0, 5, 6):
        ax.plot(np.cos(sides + z) * z,
                np.sin(sides + z) * z,
                lw=10,
                color=plt.cm.gist_yarg_r(z / 25))

    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.axis('off')
    plt.savefig("main_icon.png", format='png', bbox_inches='tight', pad_inches=0)


if __name__ == "__main__":
    generate_svg()
    generate_png()
