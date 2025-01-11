from math import cos, sin, pi, sqrt

import matplotlib.pylab as plt
import numpy as np

from tools.settings import Settings

colors = ['#f12525', '#fe6d57', '#f34126', '#f95312', '#ffe7d3']


def circle(t, r, n, z, ax):
    for x in np.linspace(-r, r, n):
        y = sqrt(r ** 2 - x ** 2)
        ax.plot(
            [x * cos(t) - y * sin(t), x * cos(t) + y * sin(t)],
            [x * sin(t) + y * cos(t), x * sin(t) - y * cos(t)],
            alpha=0.8,
            zorder=z,
            color=np.random.choice(colors),
        )


def generate(settings=Settings()):
    fig, ax = plt.subplots(figsize=(12, 12), dpi=200)
    ax = fig.add_axes((0, 0, 1, 1), facecolor="#000000")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    circle(0, 1, 60, 1, ax)
    circle1 = plt.Circle((0, 0), 0.75, color='#000000', zorder=2)
    ax.add_artist(circle1)
    circle(pi / 2, 0.75, 55, 3, ax)
    circle1 = plt.Circle((0, 0), 0.5, color='#000000', zorder=4)
    ax.add_artist(circle1)
    circle(pi / 4, 0.5, 40, 5, ax)
    circle1 = plt.Circle((0, 0), 0.25, color='#000000', zorder=6)
    ax.add_artist(circle1)
    circle(0, 0.25, 30, 7, ax)
    circle1 = plt.Circle((0, 0), 0.125, color='#000000', zorder=8)
    ax.add_artist(circle1)
    circle(pi / 2, 0.125, 30, 9, ax)
    circle1 = plt.Circle((0, 0), 0.0625, color='#000000', zorder=10)
    ax.add_artist(circle1)
    circle(pi / 4, 0.0625, 30, 11, ax)
    circle1 = plt.Circle((0, 0), 0.03125, color='#000000', zorder=12)
    ax.add_artist(circle1)
    circle(0, 0.03125, 30, 13, ax)
    plt.savefig(
        f'outputs/{settings.filename}/figure.png', facecolor='#000000'
    )
