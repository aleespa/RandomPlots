import gc
from math import cos, sin, pi, sqrt

import matplotlib.pylab as plt
import numpy as np

from common.image_processing import ImageProcessingSettings


def generate(settings: ImageProcessingSettings = None):
    settings = settings or ImageProcessingSettings(1)
    ran1 = settings.rng.beta(1.5, 1, 1000) * 40 * pi
    fig = plt.figure(figsize=(13, 13), facecolor='black', dpi=200)
    plt.axis('off')
    lin1x, lin1y = [cos(t) * t for t in np.linspace(0, 40 * pi, 1100)], [
        sin(t) * t - 10 * sqrt((cos(t) * t) ** 2 + (sin(t) * t) ** 2)
        for t in np.linspace(0, 40 * pi, 1100)
    ]
    lin2x, lin2y = [cos(t + 1) * t for t in np.linspace(0, 40 * pi, 1100)], [
        sin(t + 1) * t - 10 * sqrt((cos(t + 1) * t) ** 2 + (sin(t + 1) * t) ** 2)
        for t in np.linspace(0, 40 * pi, 1100)
    ]

    for i in range(100):
        plt.plot(
            lin1x[i * 11 : (i + 1) * 11 + 1],
            lin1y[i * 11 : (i + 1) * 11 + 1],
            lw=6,
            alpha=1,
            color='#2f7a37',
        )
        plt.plot(
            lin2x[i * 11 : (i + 1) * 11 + 1],
            lin2y[i * 11 : (i + 1) * 11 + 1],
            lw=6,
            alpha=1,
            color='#3ba347',
        )
        settings.save_to_png(fig, 'black')
    plt.close(fig)
    gc.collect()

    m = 100
    for z in np.linspace(0, 1, 30):
        plt.figure(figsize=(13, 13), facecolor='black', dpi=200)
        plt.axis('off')
        plt.plot(lin1x, lin1y, lw=6, alpha=1, color='#2f7a37')
        plt.plot(lin2x, lin2y, lw=6, alpha=1, color='#3ba347')
        plt.scatter([0], [0], s=9200 * z, marker="*", color='#f0ff00', zorder=1000)
        plt.scatter([0], [0], s=4000 * z, marker="*", color='#f6ff61', zorder=1001)
        plt.scatter([0], [0], s=2000 * z, marker="*", color='#fbffb3', zorder=1002)
        settings.save_frame('black')
        m += 1

    for i in range(100, 130):
        fig = plt.figure(figsize=(13, 13), facecolor='black', dpi=200)
        plt.axis('off')
        plt.plot(lin1x, lin1y, lw=6, alpha=1, color='#2f7a37')
        plt.plot(lin2x, lin2y, lw=6, alpha=1, color='#3ba347')
        plt.scatter([0], [0], s=9200, marker="*", color='#f0ff00', zorder=1000)
        plt.scatter([0], [0], s=4000, marker="*", color='#f6ff61', zorder=1001)
        plt.scatter([0], [0], s=2000, marker="*", color='#fbffb3', zorder=1002)
        ran1_loop = settings.rng.beta(1.5, 1, 1000) * 40 * pi
        plt.scatter(
            [cos(t) * t for t in ran1_loop],
            [sin(t) * t - 10 * sqrt((cos(t) * t) ** 2 + (sin(t) * t) ** 2) for t in ran1_loop],
            alpha=0.7,
            s=[abs(settings.rng.normal(70, 10)) for _ in ran1_loop],
            color=[plt.cm.hsv(settings.rng.uniform(0, 1)) for _ in ran1_loop],
            zorder=20,
        )
        plt.scatter(
            [cos(t) * t + 0.3 for t in ran1_loop],
            [sin(t) * t - 10 * sqrt((cos(t) * t) ** 2 + (sin(t) * t) ** 2) for t in ran1_loop],
            alpha=0.7,
            s=[abs(settings.rng.normal(10, 2)) for _ in ran1_loop],
            color='white',
            zorder=20,
            marker="*",
        )
        for _ in range(6):
            settings.save_to_png(fig, 'black')
            m += 1
        plt.close(fig)
        gc.collect()


if __name__ == '__main__':
    generate()
